import torch


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization (disabled by default).
        kwargs: Base optimizer parameters.
    """
    def __init__(self, params, base_optimizer_cls, downstream_weight=0, **kwargs):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        defaults = dict(**kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.n_weights = len(self.param_groups[0]["params"])
        if self.n_weights == 0:
            raise ValueError("No hyper-parameters to optimize.")
        self.downstream_weight = downstream_weight

    @torch.no_grad()
    def step(self, closure=None):
        """Make a single step.

        The closure is used like this: closure(target_loss_weight, *loss_weights).
        The closure must zero grads and compute gradients.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        def inner_closure():
            # Compute downstream grads.
            downstream_weight = 1
            loss_weights = [0] * self.n_weights
            closure(downstream_weight, *loss_weights)
            down_grads = self._gather_grads()

            # Compute weights grads.
            downstream_weight = 0
            weight_grads = []
            for i, w in enumerate(self.param_groups[0]["params"]):
                loss_weights[i] = 1
                closure(downstream_weight, *loss_weights)
                loss_weights[i] = 0
                loss_grads = self._gather_grads()
                assert len(down_grads) == len(loss_grads)
                product = sum([dg @ lg for dg, lg in zip(down_grads, loss_grads)])
                weight_grads.append(-product * torch.sigmoid(w) * torch.sigmoid(-w))

            # Compute model grads.
            closure(self.downstream_weight, *list(map(torch.sigmoid, self.param_groups[0]["params"])))

            # Set weights grads.
            for w, g in zip(self.param_groups[0]["params"], weight_grads):
                w.grad = g
        self.base_optimizer.step(inner_closure)

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
