import math
import torch
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR


HPO_STAGE_DOWNSTREAM = "downstream"
HPO_STAGE_FINAL = "final"


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        weights_parametrization: Either "linear", "exp", "softplus", "tanh", "abs", or "sigmoid".
        weights_normalization: Whether to normalize weights by their sum or not ("sum", "norm", "max", or "none").
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        closed_form: If true, set weights without gradient optimization.
        closed_form_momentum: Use momentum when closed form is used.
        clip_hp_grad: Clipping value for hyperparameters gradients.
        kwargs: Base optimizer parameters.

    Example usage:
    ```
    optimizer = CorrHPOptimizer([{"params": [w1, w2]},  # Weights for tuning.
                                 {"params": model.parameters()}],
                                torch.optim.Adam,
                                lr=0.01)

    output = model(x)
    down_loss, loss1, loss2 = criterion(x)

    def closure(down_weight, free_term_weight, w1, w2, stage=None):
        optimizer.zero_grad()
        loss = down_weight * down_loss + w1 * loss1 + w2 * loss2 + free_term_weight * loss_free
        loss.backward(retain_graph=stage != HPO_STAGE_FINAL)

    optimizer.hpo_step(closure)
    ```

    NOTE: Free term is responsible for the part of the loss that must to be tuned.
    """
    def __init__(self, params, base_optimizer_cls, downstream_weight="merge",
                 weights_parametrization="sigmoid", weights_normalization="sum",
                 apply_optimizer_correction=False, closed_form=False, closed_form_momentum=0,
                 normalize_down_grad=False,
                 clip_hp_grad=None, eps=1e-6, **kwargs):
        params = list(params)
        if closed_form:
            weights_parametrization = "linear"
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        if weights_parametrization not in ["linear", "abs", "tanh", "sigmoid", "softplus", "exp"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["sum", "norm", "max", "none"]:
            raise ValueError(f"Unknown weights normalization method: {weights_normalization}")
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
        self.apply_optimizer_correction = apply_optimizer_correction
        self.closed_form = closed_form
        self.closed_form_momentum = closed_form_momentum
        self.normalize_down_grad = normalize_down_grad
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps
        self._down_grads_cache = None

    def step(self, closure, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        return self.base_optimizer.step(closure)

    def cache_downstream(self, closure=None):
        """Cache downstream gradient for future computations.

        Use this method to tune hyperparamers on validation batches.
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        downstream_weight = 1
        loss_weights = [0] * self.n_weights
        closure(downstream_weight, 0, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
        self._down_grads_cache = self._gather_grads(normalize=self.normalize_down_grad)

    def hpo_step(self, closure=None):
        """Make a single step.

        The closure is used like this: closure(target_loss_weight, free_term_weight, *loss_weights, stage=None).
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
            elif self.weights_parametrization == "tanh":
                weights = torch.tanh(logits)
            elif self.weights_parametrization == "abs":
                weights = torch.abs(logits)
            elif self.weights_parametrization == "softplus":
                weights = torch.nn.functional.softplus(logits)
            else:
                weights = logits
                assert self.weights_parametrization == "linear"

            # Compute downstream grads.
            downstream_weight = 1
            loss_weights = [0] * self.n_weights
            closure(downstream_weight, 0, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
            down_train_grads = self._gather_grads(normalize=self.normalize_down_grad)
            if self._down_grads_cache is None:
                down_grads = down_train_grads
            else:
                down_grads = self._down_grads_cache

            # Compute weights grads.
            downstream_weight = 0
            weight_grads = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            compute_grad_sum = (not self.closed_form) and (self.weights_normalization not in {"none", "max"})
            if compute_grad_sum:
                grad_sum = [torch.zeros_like(v) for v in down_grads]
            compute_norms = self.closed_form
            if compute_norms:
                norms = torch.zeros_like(weights)
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                closure(downstream_weight, 0, *loss_weights, stage=i)
                loss_weights[i] = 0
                loss_grads = self._gather_grads(apply_optimizer_correction=self.apply_optimizer_correction)
                assert len(down_grads) == len(loss_grads)
                product = sum([dg @ lg for dg, lg in zip(down_grads, loss_grads)])
                weight_grads[i] -= product
                if compute_grad_sum:
                    for j, v in enumerate(loss_grads):
                        grad_sum[j] += v * w
                if compute_norms:
                    norms[i] = torch.stack([g.square().sum() for g in loss_grads]).sum().sqrt()

            if self.closed_form:
                weight_grads = -weight_grads / norms[i].square()
                if self.weights_normalization == "sum":
                    s = weight_grads.sum() + self.eps
                    weight_grads /= s
                elif self.weights_normalization == "norm":
                    norm = torch.linalg.norm(weight_grads) + self.eps
                    weight_grads /= norm
                elif self.weights_normalization == "max":
                    norm = weight_grads.max()
                    weight_grads /= norm
                else:
                    assert self.weights_normalization == "none"
                actual_weights = weights * self.closed_form_momentum + weight_grads * (1 - self.closed_form_momentum)
                del weight_grads
            else:
                if self.weights_normalization == "sum":
                    s = weights.sum() + self.eps
                    weight_grads /= s
                    product = sum([dg @ lg for dg, lg in zip(down_grads, grad_sum)])
                    weight_grads += product / s ** 2
                    weight_grads *= len(weights)  # Scale by the number of weights.
                    actual_weights = weights / s
                elif self.weights_normalization == "norm":
                    norm = torch.linalg.norm(weights) + self.eps
                    weight_grads /= norm
                    product = sum([dg @ lg for dg, lg in zip(down_grads, grad_sum)])
                    weight_grads += weights * product / norm ** 3
                    weight_grads *= math.sqrt(len(weights))  # Scale by the norm of union vector.
                    actual_weights = weights / norm
                elif self.weights_normalization == "max":
                    norm, i = weights.max(0)
                    weight_grads[i] = 0
                    weight_grads /= norm
                    actual_weights = weights / norm
                else:
                    assert self.weights_normalization == "none"
                    actual_weights = weights

            if self.weights_parametrization == "sigmoid":
                weight_grads *= weights * torch.sigmoid(-logits)
            elif self.weights_parametrization == "tanh":
                weight_grads /= torch.cosh(logits).square() + self.eps
            elif self.weights_parametrization == "abs":
                weight_grads = torch.where(logits >= 0, weight_grads, -weight_grads)  # torch.sign freezes at zero.
            elif self.weights_parametrization == "exp":
                weight_grads *= weights
            elif  self.weights_parametrization == "softplus":
                weight_grads *= torch.sigmoid(logits)
            else:
                assert self.weights_parametrization == "linear"
            if self.clip_hp_grad is not None:
                grad_norm = torch.linalg.norm(weight_grads)
                if grad_norm > self.clip_hp_grad:
                    weight_grads *= self.clip_hp_grad / grad_norm

            # Compute model grads.
            closure(self.downstream_weight, 1, *actual_weights, stage=HPO_STAGE_FINAL)

            if self.closed_form:
                for w, v in zip(self.param_groups[0]["params"], actual_weights):
                    w.data.copy_(v)
                weight_grads = [None] * self.n_weights

            # Set weights grads.
            for w, g in zip(self.param_groups[0]["params"], weight_grads):
                w.grad = g

            if self.downstream_merge:
                i = 0
                for group in self.param_groups[1:]:
                    for p in group["params"]:
                        if p.grad is None:
                            p.grad = down_train_grads[i].reshape(p.shape)
                        else:
                            p.grad = torch.where(p.grad.abs() > 0, p.grad, down_train_grads[i].reshape(p.shape))
                        i += 1
                assert i == len(down_train_grads)

        return self.step(inner_closure, inner=True)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _gather_grads(self, apply_optimizer_correction=False, normalize=False):
        grads = []
        for group in self.param_groups[1:]:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
                    p.grad = None
        if apply_optimizer_correction:
            # We don't pass gradient to the velocity vector for simplicity.
            if isinstance(self.base_optimizer, torch.optim.Adam):
                i = 0
                for group in self.param_groups[1:]:
                    _, beta2 = group["betas"]
                    eps = group["eps"]
                    for p in group["params"]:
                        state = self.base_optimizer.state[p]
                        exp_avg_sq = state.get("exp_avg_sq", None)
                        if exp_avg_sq is None:
                            i += 1
                            continue
                        step = state["step"]
                        bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5
                        grads[i] /= exp_avg_sq.sqrt().flatten() / bias_correction2_sqrt + eps
                        i += 1
                assert i == len(grads)
            elif isinstance(self.base_optimizer, torch.optim.SGD):
                pass  # No need for correction.
            else:
                raise NotImplementedError(f"Can't apply correction to {type(self.base_optimizer).__name__}")
        if normalize:
            norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt() + self.eps
            grads = [g / norm for g in grads]
        return grads


class RepetitiveWarmupScheduler(LambdaLR):
    def __init__(self, optimizer, cycle_steps, warmup_steps,
                 reset_optimizer=True, last_epoch=-1,
                 **kwargs):  # ignored, for configuration simplicity.
        self._cycle_steps = cycle_steps
        self._warmup_steps = warmup_steps
        self._reset_optimizer = reset_optimizer
        self._init_optimizer_state = deepcopy(optimizer.state_dict())
        self.default_interval = "step"
        super().__init__(optimizer, last_epoch=last_epoch,
                         lr_lambda=self._lr_lambda_impl)

    def state_dict(self):
        state = super().state_dict()
        state["init_optimizer_state"] = self._init_optimizer_state
        return state

    def load_state_dict(self, state):
        self._init_optimizer_state = state.pop("init_optimizer_state")
        assert self._init_optimizer_state is not None
        return super().load_state_dict(state)

    def _lr_lambda_impl(self, epoch):
        in_cycle = epoch % self._cycle_steps
        if in_cycle == 0 and self._reset_optimizer:
            self.optimizer.load_state_dict(deepcopy(self._init_optimizer_state))
        return 0 if in_cycle < self._warmup_steps else 1
