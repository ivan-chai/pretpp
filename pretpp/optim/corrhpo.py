import torch


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        normalize_weights: Either "sigmoid", "softmax", or "none"
        kwargs: Base optimizer parameters.
    """
    def __init__(self, params, base_optimizer_cls, downstream_weight="merge", normalize_weights="softmax", eps=1e-6, **kwargs):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        if normalize_weights not in ["sigmoid", "softmax", "none"]:
            raise ValueError(f"Unknown weights normalization method: {normalize_weights}")
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
        self.normalize_weights = normalize_weights
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
            if self.normalize_weights == "sigmoid":
                weights = torch.sigmoid(logits)
                normalize = False
            elif self.normalize_weights == "softmax":
                weights = torch.exp(logits)
                normalize = True
            else:
                assert self.normalize_weights == "none"
                weights = logits
                normalize = False

            # Compute downstream grads.
            downstream_weight = 1
            loss_weights = [0] * self.n_weights
            closure(downstream_weight, *loss_weights, final=False)
            down_grads = self._gather_grads()

            # Compute weights grads.
            downstream_weight = 0
            weight_grads = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            if normalize:
                grad_sum = [torch.zeros_like(v) for v in down_grads]
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                closure(downstream_weight, *loss_weights, final=False)
                loss_weights[i] = 0
                loss_grads = self._gather_grads()
                assert len(down_grads) == len(loss_grads)
                product = sum([dg @ lg for dg, lg in zip(down_grads, loss_grads)])
                weight_grads[i] -= product
                if normalize:
                    for j, v in enumerate(loss_grads):
                        grad_sum[j] += v * w

            if normalize:
                s = weights.sum() + self.eps
                weight_grads /= s
                product = sum([dg @ lg for dg, lg in zip(down_grads, grad_sum)])
                weight_grads += product / s ** 2

            if self.normalize_weights == "sigmoid":
                for i, (w, w_logit) in enumerate(zip(weights, logits)):
                    weight_grads[i] *= w * torch.sigmoid(-w_logit)
            elif self.normalize_weights == "softmax":
                weight_grads *= weights
            else:
                assert self.normalize_weights == "none"
            # Compute model grads.
            closure(self.downstream_weight, *list(map(torch.sigmoid, self.param_groups[0]["params"])), final=True)

            # Set weights grads.
            for w, g in zip(self.param_groups[0]["params"], weight_grads):
                w.grad = g

            if self.downstream_merge:
                i = 0
                for group in self.param_groups[1:]:
                    for p in group["params"]:
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
