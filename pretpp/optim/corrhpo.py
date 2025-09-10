import torch


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        weights_parametrization: Either "exp", "softplus", or "sigmoid".
        weights_normalization: Whether to normalize weights by their sum or not.
        clip_hp_grad: Clipping value for hyperparameters gradients.
        kwargs: Base optimizer parameters.
    """
    def __init__(self, params, base_optimizer_cls, downstream_weight="merge",
                 weights_parametrization="sigmoid", weights_normalization=True,
                 clip_hp_grad=None, eps=1e-6, **kwargs):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        if weights_parametrization not in ["sigmoid", "softplus", "exp"]:
            raise ValueError(f"Unknown weights normalization method: {weights_parametrization}")
        defaults = dict(**kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.n_weights = len(self.param_groups[0]["params"])
        if self.n_weights == 0:
            raise ValueError("No hyper-parameters to optimize.")
        if downstream_weight == "merge":
            self.downstream_merge = True
            self.downstream_weight = 0
        else:
            self.downstream_merge = False
            self.downstream_weight = float(downstream_weight)
        self.weights_parametrization = weights_parametrization
        self.weights_normalization = weights_normalization
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps

    def step(self, closure, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        return self.base_optimizer.step(closure)

    def hpo_step(self, closure=None):
        """Make a single step.

        The closure is used like this: closure(target_loss_weight, *loss_weights).
        The closure must zero grads and compute gradients.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        @torch.no_grad()
        def inner_closure():
            logits = torch.stack(self.param_groups[0]["params"])
            if self.weights_parametrization == "exp":
                weights = torch.exp(logits)
            elif self.weights_parametrization == "sigmoid":
                weights = torch.sigmoid(logits)
            else:
                assert self.weights_parametrization == "softplus"
                weights = torch.nn.functional.softplus(logits)

            # Compute downstream grads.
            downstream_weight = 1
            loss_weights = [0] * self.n_weights
            closure(downstream_weight, *loss_weights, final=False)
            down_grads = self._gather_grads()

            # Compute weights grads.
            downstream_weight = 0
            weight_grads = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            if self.weights_normalization:
                grad_sum = [torch.zeros_like(v) for v in down_grads]
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                closure(downstream_weight, *loss_weights, final=False)
                loss_weights[i] = 0
                loss_grads = self._gather_grads()
                assert len(down_grads) == len(loss_grads)
                product = sum([dg @ lg for dg, lg in zip(down_grads, loss_grads)])
                weight_grads[i] -= product
                if self.weights_normalization:
                    for j, v in enumerate(loss_grads):
                        grad_sum[j] += v * w

            if self.weights_normalization:
                s = weights.sum() + self.eps
                weight_grads /= s
                product = sum([dg @ lg for dg, lg in zip(down_grads, grad_sum)])
                weight_grads += product / s ** 2

            if self.weights_parametrization == "sigmoid":
                weight_grads *= weights * torch.sigmoid(-logits)
            elif self.weights_parametrization == "exp":
                weight_grads *= weights
            else:
                assert self.weights_parametrization == "softplus"
                weight_grads *= torch.sigmoid(logits)
            if self.clip_hp_grad is not None:
                grad_norm = torch.linalg.norm(weight_grads)
                if grad_norm > self.clip_hp_grad:
                    weight_grads *= self.clip_hp_grad / grad_norm

            # Compute model grads.
            actual_weights = weights / (weights.sum() + self.eps) if self.weights_normalization else weights
            closure(self.downstream_weight, *actual_weights, final=True)

            # Set weights grads.
            for w, g in zip(self.param_groups[0]["params"], weight_grads):
                w.grad = g

            if self.downstream_merge:
                i = 0
                for group in self.param_groups[1:]:
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = down_grads[i].reshape(p.shape)
                        else:
                            p.grad = torch.where(p.grad.abs() > 0, p.grad, down_grads[i].reshape(p.shape))
                        i += 1
                assert i == len(down_grads)

        return self.step(inner_closure, inner=True)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _gather_grads(self):
        grads = []
        for group in self.param_groups[1:]:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
                    p.grad = None
        return grads
