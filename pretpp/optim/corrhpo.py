import math
import numpy as np
import torch
from copy import deepcopy
from numbers import Number
from torch.optim.lr_scheduler import LambdaLR

from .solvers import closed_form, closed_form_spherical, closed_form_fw, closed_form_trmse, ProcrustesSolver


HPO_STAGE_DOWNSTREAM = "downstream"
HPO_STAGE_FINAL = "final"

HPO_ERROR = "error"
HPO_WEIGHTS = "weights"


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        weights_parametrization: Either "linear", "tanh", "abs", or "sigmoid".
        weights_normalization: Whether to normalize weights by their sum or not ("sum", "norm", or "none").
        algorithm: Either "sgd", "closed-form[-sphere]", "closed-form-trmse[-proj]", "closed-form-mse", "closed-form-pos-uni", "closed-form-nearest", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be dictionary with "main" and "downstream" keys
            for the main and downstream losses respectively. An additional "weights" key can be provided to control
            weights smoothing in closed-form algorithms.
        ema_interleaved: Update gradients one at a step.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        mtl: Apply gradients correction from multi-task learning (either False/None or "amtl").
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

    def closure(down_weight, w1, w2, stage=None):
        optimizer.zero_grad()
        loss = down_weight * down_loss + w1 * loss1 + w2 * loss2
        loss.backward(retain_graph=stage != HPO_STAGE_FINAL)

    optimizer.hpo_step(closure)
    ```
    """
    def __init__(self, params, base_optimizer_cls, downstream_weight="merge",
                 weights_parametrization="abs", weights_normalization="norm",
                 algorithm="closed-form-sphere", ema=0, ema_interleaved=False,
                 normalize_down_grad=False, apply_optimizer_correction=False,
                 mtl=None, clip_hp_grad=None, eps=1e-6, **kwargs):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        if algorithm not in {"sgd", "closed-form", "closed-form-sphere", "closed-form-trmse", "closed-form-trmse-proj", "closed-form-mse", "closed-form-pos-uni", "closed-form-nearest", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if algorithm.startswith("closed-form") and (weights_parametrization not in {"linear", "abs"}):
            raise ValueError("Closed-form approach is compatible with linear and abs parametrizations only.")
        if weights_parametrization not in ["linear", "abs", "tanh", "sigmoid"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["sum", "norm", "none"]:
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
        self.algorithm = algorithm
        if isinstance(ema, Number):
            ema = {k: ema for k in ["main", "downstream"]}
        if ("main" not in ema) or ("downstream" not in ema):
            raise ValueError(f"ema: expected dictionary with 'main', 'downstream', and optional 'weights' keys.")
        if "weights" not in ema:
            ema["weights"] = ema["main"]
        self.main_momentum = ema["main"]
        self.downstream_momentum = ema["downstream"]
        self.weights_momentum = ema["weights"]
        self.ema_interleaved = ema_interleaved

        self.normalize_down_grad = normalize_down_grad
        self.apply_optimizer_correction = apply_optimizer_correction
        self.mtl = mtl
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps

        # todo: use optimizer state for gradient caches.
        self._ema_step = 0
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None} | {i: None for i in range(self.n_weights)}
        if algorithm.startswith("closed-form") and (algorithm not in {"closed-form-mse", "closed-form-pos-uni", "closed-form-nearest"}):
            if not algorithm.startswith("closed-form-trmse"):
                self._grads_cache[HPO_ERROR] = None
            self._grads_cache.update({f"cov_{k}": None for k in self._grads_cache})
        if algorithm.startswith("closed-form"):
            self._grads_cache[HPO_WEIGHTS] = None

    def step(self, closure, inner=False):
        if not inner:
            raise valueerror("please, use 'hpo_step' function.")
        return self.base_optimizer.step(closure)

    def _update_grads_cache(self, grads, stage=None):
        assert stage in self._grads_cache
        if stage == HPO_STAGE_DOWNSTREAM:
            momentum = self.downstream_momentum
        elif stage == HPO_WEIGHTS:
            momentum = self.weights_momentum
        else:
            momentum = self.main_momentum
        # Means.
        if (self._grads_cache[stage] is not None) and momentum:
            self._grads_cache[stage] = self._grads_cache[stage] * momentum + grads * (1 - momentum)
        else:
            self._grads_cache[stage] = grads
        # Covs.
        if self.algorithm.startswith("closed-form") and (self.algorithm not in {"closed-form-mse", "closed-form-pos-uni", "closed-form-nearest"}) and (stage != HPO_WEIGHTS):
            delta_sq = (grads - self._grads_cache[stage]).square()
            k = f"cov_{stage}"
            if (self._grads_cache[k] is not None) and momentum:
                self._grads_cache[k] = self._grads_cache[k] * momentum + delta_sq * (1 - momentum)
            else:
                self._grads_cache[k] = delta_sq
        return self._grads_cache[stage]

    def cache_downstream(self, closure=None):
        """cache downstream gradient for future computations.

        use this method to tune hyperparamers on validation batches.
        """
        assert closure is not None, "need closure"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        downstream_weight = 1
        loss_weights = [0] * self.n_weights
        closure(downstream_weight, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
        self._update_grads_cache(self._gather_grads(normalize=self.normalize_down_grad), stage=HPO_STAGE_DOWNSTREAM)

    def remove_cache(self, stage=None):
        if stage is None:
            self._grads_cache = {name: None for name in self._grads_cache}
        else:
            self._grads_cache[stage] = None
            if self.algorithm.startswith("closed-form") and (self.algorithm not in {"closed-form-mse", "closed-form-pos-uni", "closed-form-nearest"}) and (stage != HPO_WEIGHTS):
                self._grads_cache[f"cov_{stage}"]= None

    @property
    def metrics(self):
        result = {}
        if self.algorithm.startswith("closed-form") and (self.algorithm not in {"closed-form-mse", "closed-form-pos-uni", "closed-form-nearest"}):
            for k, v in self._grads_cache.items():
                if not isinstance(k, int) and k.startswith("cov_") and (v is not None):
                    result[k] = v.mean().item()
            if self._grads_cache.get(HPO_ERROR, None) is not None:
                result["hpo_error"] = torch.linalg.norm(self._grads_cache[HPO_ERROR])
        return result

    def _get_scale_norm(self, weights):
        if self.weights_normalization == "sum":
            scale = self.n_weights
            norm = weights.sum() + self.eps
        elif self.weights_normalization == "norm":
            scale = math.sqrt(self.n_weights)
            norm = torch.linalg.norm(weights) + self.eps
        else:
            assert self.weights_normalization == "none"
            scale = 1
            norm = 1
        return scale, norm

    def hpo_step(self, closure=None, use_cached_downstream=False):
        """Make a single step.

        Args:
            use_cached_downstream: Use gradients cache, don't recompute downstream gradients.

        The closure is used like this: closure(target_loss_weight, *loss_weights, stage=None).
        The closure must zero grads and compute gradients.
        """
        assert closure is not None, "Need closure"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        @torch.no_grad()
        def inner_closure():
            logits = torch.stack(self.param_groups[0]["params"])
            if self.weights_parametrization == "sigmoid":
                weights = torch.sigmoid(logits)
            elif self.weights_parametrization == "tanh":
                weights = torch.tanh(logits)
            elif self.weights_parametrization == "abs":
                weights = torch.abs(logits)
            else:
                weights = logits
                assert self.weights_parametrization == "linear"

            # Compute downstream grads.
            loss_weights = [0] * self.n_weights
            if (not use_cached_downstream) or self.downstream_merge:
                downstream_weight = 1
                closure(downstream_weight, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
                down_train_grads = self._gather_grads(normalize=self.normalize_down_grad)
                if not use_cached_downstream:
                    self._update_grads_cache(down_train_grads, stage=HPO_STAGE_DOWNSTREAM)
            if use_cached_downstream or self.downstream_momentum:
                down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            else:
                down_grads = down_train_grads
            assert down_grads is not None

            # Caches for normalization differentiation.
            compute_products = self.algorithm in {"sgd"}
            if compute_products:
                products = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            compute_grad_sum = (self.algorithm == "sgd") and (self.weights_normalization in {"sum", "norm"})
            if compute_grad_sum:
                grad_sum = torch.zeros_like(down_grads)
            store_all_grads = self.algorithm.startswith("closed-form")
            if store_all_grads:
                all_grads = []

            # Compute weights grads.
            downstream_weight = 0
            for i, w in enumerate(weights):
                if self.ema_interleaved and (self._ema_step % len(weights) != i) and (self._grads_cache[i] is not None):
                    loss_grads = self._grads_cache[i]
                else:
                    loss_weights[i] = 1
                    closure(downstream_weight, *loss_weights, stage=i)
                    loss_weights[i] = 0
                    loss_grads = self._gather_grads(apply_optimizer_correction=self.apply_optimizer_correction)
                    loss_grads = self._update_grads_cache(loss_grads, stage=i)
                if compute_products:
                    products[i] = down_grads @ loss_grads
                if compute_grad_sum:
                    grad_sum += loss_grads * w
                if store_all_grads:
                    all_grads.append(loss_grads)

            if store_all_grads:
                if self.mtl is not None:
                    assert self.mtl == "amtl"
                    all_grads = torch.stack(all_grads, 1)  # (P, W).
                    all_grads, _, _ = ProcrustesSolver.apply(all_grads[None], False)
                    all_grads = all_grads[0].T  # (W, P).
                else:
                    all_grads = torch.stack(all_grads, 0)  # (W, P).
            else:
                assert not self.mtl

            if self.algorithm == "sgd":
                scale, norm = self._get_scale_norm(weights)
                actual_weights = scale * weights / norm
                if self.weights_normalization == "sum":
                    product = down_grads @ grad_sum
                    weight_grads = -products / norm + product / norm ** 2
                    weight_grads *= scale  # Scale by the number of weights.
                elif self.weights_normalization == "norm":
                    product = down_grads @ grad_sum
                    weight_grads = -products / norm + weights * product / (norm ** 3)
                    weight_grads *= scale  # Scale by the norm of union vector.
                else:
                    assert self.weights_normalization == "none"
                    weight_grads = -products

                if self.weights_parametrization == "sigmoid":
                    weight_grads *= weights * torch.sigmoid(-logits)
                elif self.weights_parametrization == "tanh":
                    weight_grads /= torch.cosh(logits).square() + self.eps
                elif self.weights_parametrization == "abs":
                    weight_grads = torch.where(logits >= 0, weight_grads, -weight_grads)  # torch.sign freezes at zero.
                else:
                    assert self.weights_parametrization == "linear"
                if self.clip_hp_grad is not None:
                    grad_norm = torch.linalg.norm(weight_grads)
                    if grad_norm > self.clip_hp_grad:
                        weight_grads *= self.clip_hp_grad / (grad_norm + self.eps)
            elif self.algorithm in {"closed-form", "closed-form-sphere"}:
                if self.algorithm == "closed-form-sphere":
                    spherical = True
                else:
                    assert self.algorithm == "closed-form"
                    spherical = False
                if self.weights_parametrization == "abs":
                    positive = True
                else:
                    assert self.weights_parametrization == "linear"
                    positive = False
                if self.weights_normalization == "norm":
                    unit_norm = True
                else:
                    assert self.weights_normalization == "none"
                    unit_norm = False
                dim = len(down_grads)

                # Gather and normalize.
                all_grads_covs = [self._grads_cache[f"cov_{i}"] for i in range(self.n_weights)]
                if any([c is None for c in all_grads_covs]):
                    all_grads_covs = torch.zeros_like(all_grads)  # (W, P).
                else:
                    all_grads_covs = torch.stack(all_grads_covs)  # (W, P).

                if self.normalize_down_grad:
                    target_norm = torch.linalg.norm(down_grads) + self.eps
                    target = down_grads / target_norm
                else:
                    target_norm = torch.ones_like(down_grads[0])
                    target = down_grads
                target_cov = self._grads_cache[f"cov_{HPO_STAGE_DOWNSTREAM}"]
                if target_cov is None:
                    target_cov = torch.zeros_like(target)
                else:
                    target_cov = target_cov / (target_norm.square() + self.eps)

                err_cov = self._grads_cache[f"cov_{HPO_ERROR}"]
                if err_cov is None:
                    closest = torch.linalg.norm(all_grads - target, dim=1).argmin()
                    err_cov = (all_grads[closest] - target).square()  # (P).

                if spherical:
                    actual_weights = closed_form_spherical(all_grads, target,
                                                           covs=all_grads_covs.mean(1),  # (W).
                                                           cov_err=target_cov.mean() + err_cov.mean(),
                                                           positive=positive,
                                                           eps=self.eps)
                else:
                    if not unit_norm:
                        raise NotImplementedError("Either unit-norm or spherical algorithm must be used")
                    # Approximation.
                    all_grads_covs = all_grads_covs.mean(0)  # (P).
                    actual_weights = closed_form(all_grads, target,
                                                 covs=all_grads_covs + target_cov + err_cov,
                                                 positive=positive,
                                                 eps=self.eps)
                    # Scale weights to reduce error.
                    # TODO: use covariances?
                    solution = actual_weights @ all_grads
                    weights_scale = (solution @ target) / (solution @ solution + self.eps)
                    if positive and weights_scale < 0:
                        weights_scale = 0
                    actual_weights *= weights_scale

                # Update error stats.
                error = target - actual_weights @ all_grads  # (P).
                err_cov_new = error.square()
                key = HPO_ERROR
                cov_key = f"cov_{HPO_ERROR}"
                if (self._grads_cache[key] is not None) and self.main_momentum:
                    self._grads_cache[key] = self._grads_cache[key] * self.main_momentum + error * (1 - self.main_momentum)
                    self._grads_cache[cov_key] = self._grads_cache[cov_key] * self.main_momentum + err_cov_new * (1 - self.main_momentum)
                else:
                    self._grads_cache[key] = error
                    self._grads_cache[cov_key] = err_cov_new

                # Apply scaling.
                if self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights) / (torch.linalg.norm(actual_weights) + self.eps)
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                actual_weights = scale * actual_weights
            elif self.algorithm == "closed-form-mse":
                if self.weights_parametrization == "abs":
                    positive = True
                else:
                    assert self.weights_parametrization == "linear"
                    positive = False
                dim = len(down_grads)

                # Gather and normalize.
                target = down_grads

                actual_weights = closed_form_trmse(all_grads, target,
                                                   positive=positive,
                                                   eps=self.eps)

                if torch.linalg.norm(actual_weights) < self.eps:
                    actual_weights = torch.ones_like(actual_weights)

                # Apply scaling.
                if self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights) / (torch.linalg.norm(actual_weights) + self.eps)
                elif self.weights_normalization == "sum":
                    scale = self.n_weights / (actual_weights.sum() + self.eps)
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                actual_weights = scale * actual_weights
            elif self.algorithm.startswith("closed-form-trmse"):
                if self.algorithm == "closed-form-trmse":
                    proj = False
                else:
                    assert self.algorithm == "closed-form-trmse-proj"
                    proj = True
                if self.weights_parametrization == "abs":
                    positive = True
                else:
                    assert self.weights_parametrization == "linear"
                    positive = False
                if self.weights_normalization == "norm":
                    unit_norm = True
                else:
                    assert self.weights_normalization == "none"
                    unit_norm = False
                dim = len(down_grads)

                # Gather and normalize.
                all_grads_covs = [self._grads_cache[f"cov_{i}"] for i in range(self.n_weights)]
                if any([c is None for c in all_grads_covs]):
                    all_grads_covs = torch.zeros_like(all_grads)  # (W, P).
                else:
                    all_grads_covs = torch.stack(all_grads_covs)  # (W, P).

                if self.normalize_down_grad:
                    target_norm = torch.linalg.norm(down_grads) + self.eps
                    target = down_grads / target_norm
                else:
                    target_norm = torch.ones_like(down_grads[0])
                    target = down_grads

                if proj:
                    scale = target / (target.mean() + self.eps)  # (P).
                    all_grads *= scale
                    target = target * scale
                    all_grads_covs *= scale.square()

                actual_weights = closed_form_trmse(all_grads, target,
                                                   covs=all_grads_covs,
                                                   positive=positive,
                                                   eps=self.eps)

                if torch.linalg.norm(actual_weights) < self.eps:
                    actual_weights = torch.ones_like(actual_weights)

                # Apply scaling.
                if self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights) / (torch.linalg.norm(actual_weights) + self.eps)
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                actual_weights = scale * actual_weights
            elif self.algorithm == "closed-form-pos-uni":
                if self.weights_parametrization != "abs":
                    raise NotImplementedError(f"Pos-uni with {self.weights_parametrization} parametrization.")
                dim = len(down_grads)

                # Gather and normalize.
                target = down_grads
                covs = all_grads @ target  # (W).
                positive = covs > 0
                if positive.sum() == 0:
                    actual_weights = torch.ones_like(covs)
                else:
                    actual_weights = positive.float()

                # Apply scaling.
                if self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights) / (torch.linalg.norm(actual_weights) + self.eps)
                elif self.weights_normalization == "sum":
                    scale = self.n_weights / (actual_weights.sum() + self.eps)
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                actual_weights = scale * actual_weights
            elif self.algorithm == "closed-form-nearest":
                if self.weights_parametrization != "abs":
                    raise NotImplementedError(f"Nearest with {self.weights_parametrization} parametrization.")
                dim = len(down_grads)

                # Gather and normalize.
                target = down_grads
                covs = all_grads @ target  # (W).
                actual_weights = torch.nn.functional.one_hot(torch.argmax(covs), self.n_weights).float()

                # Apply scaling.
                if self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights) / (torch.linalg.norm(actual_weights) + self.eps)
                elif self.weights_normalization == "sum":
                    scale = self.n_weights / (actual_weights.sum() + self.eps)
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                actual_weights = scale * actual_weights
            else:
                assert self.algorithm == "none"
                scale, norm = self._get_scale_norm(weights)
                actual_weights = scale * weights / norm

            if self.algorithm.startswith("closed-form") and self.weights_momentum:
                actual_weights = self._update_grads_cache(actual_weights, stage=HPO_WEIGHTS)

            # Compute model grads.
            downstream_weight = self.downstream_weight
            closure(self.downstream_weight, *actual_weights, stage=HPO_STAGE_FINAL)

            if self.algorithm.startswith("closed-form"):
                for w, v in zip(self.param_groups[0]["params"], actual_weights):
                    w.data.copy_(v)
                    w.grad = None
            elif self.algorithm == "sgd":
                for w, g in zip(self.param_groups[0]["params"], weight_grads):
                    w.grad = g
            else:
                assert self.algorithm == "none"
                for w, v in zip(self.param_groups[0]["params"], actual_weights):
                    w.grad = None

            if self.downstream_merge:
                offset = 0
                for group in self.param_groups[1:]:
                    for p in group["params"]:
                        numel = p.numel()
                        down_p_grad = down_train_grads[offset:offset + numel].reshape(p.shape)
                        if p.grad is None:
                            p.grad = down_p_grad
                        else:
                            p.grad = torch.where(p.grad.abs() > 0, p.grad, down_p_grad)
                        offset += numel
                assert offset == len(down_train_grads)

        return self.step(inner_closure, inner=True)

    def state_dict(self):
        state = super().state_dict()
        state["grads_cache"] = dict(self._grads_cache)
        state["ema_step"] = self._ema_step
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        p = self.param_groups[0]["params"][0]
        self._grads_cache.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                  for k, v in state_dict.get("grads_cache", {}).items()})
        self._ema_step = state_dict["ema_step"]

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
        grads = torch.cat(grads)
        if normalize:
            grads /= torch.linalg.norm(grads)
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
