import math
import numpy as np
import torch
from copy import deepcopy
from functools import partial
from numbers import Number
from torch.optim.lr_scheduler import LambdaLR
from scipy.optimize import bisect, minimize


HPO_STAGE_DOWNSTREAM = "downstream"
HPO_STAGE_FREE = "free"
HPO_STAGE_FINAL = "final"


def _find_lambda_closest_unit_norm(cov, b, eps=1e-6, steps=100):
    """Solve ||inv(C + x I) b|| = 1."""
    dim = len(cov)
    s, u = np.linalg.eigh(cov)  # cov = u @ diag(s) @ u.T, u - orthonormal.
    beta_sq = (u.T @ b) ** 2
    s_min = s.min()

    # Select non-zero terms.
    mask = beta_sq > 0
    beta_sq = beta_sq[mask]
    s = s[mask]
    if len(s) == 0:
        return 0

    func = lambda x: (beta_sq / (s + x) ** 2).sum() - 1
    start = -s_min + eps
    if func(start) < 0:
        x = -s_min
    else:
        end_offset = 1
        while func(start + end_offset) > -eps:
            end_offset *= 2
        end = start + end_offset
        x = bisect(func, start, end, maxiter=steps, xtol=eps)
    return x


def find_closest_unit_norm(basis, target, eps=1e-6, steps=100):
    zero_mask = torch.linalg.norm(basis, dim=1) < eps
    zero_index = zero_mask.nonzero()[0, 0].item() if zero_mask.any() else None

    cov = (basis @ basis.T).cpu().double().numpy()  # (W, W).
    b = (basis @ target).cpu().double().numpy()  # (W).
    l = _find_lambda_closest_unit_norm(cov, b)
    reg = np.eye(len(cov))
    weights = np.linalg.lstsq(cov + l * reg, b)[0]

    if zero_index is not None:
        assert abs(weights[zero_index]) < eps
        weights[zero_index] = np.sqrt(max(0, 1 - np.sum(weights ** 2)))
    return torch.from_numpy(weights).to(device=basis.device)


def quadratic_program_positive(C, b, steps=100,
                               method="SLSQP"):  #SLSQP.
    """Minimize 0.5 x^T C x + b^T x with positive x."""
    device = C.device
    dim = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()
    func = lambda x: 0.5 * x.T @ C @ x + b @ x
    jac = lambda x: C @ x + b
    hess = lambda x: C
    eye = np.eye(dim)
    cons = {
        "type": "ineq",
        "fun": lambda x: x,
        "jac": lambda x: eye
    }
    opt = {
        "disp": False,
        "maxiter": steps
    }
    x0 = np.ones(dim) / dim
    weights = minimize(func, x0, jac=jac, constraints=cons,
                       method=method, options=opt)["x"]
    weights = np.clip(weights, a_min=0, a_max=None)
    return torch.from_numpy(weights).to(device)


def normal_ce_solve(C, b, r, covs, eps=1e-6, positive=False):
    """Minimize log s^2 + |w^T @ C @ w - 2 (w, b) + r| / d / s^2, where s^2 = sum w^2_i s^2_i."""
    device = C.device
    dim = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()
    r = r.item()
    covs = covs.cpu().double().numpy()
    def func(x):
        s2 = ((x ** 2) * covs).sum() + eps
        delta_norm_sq = x.T @ C @ x - 2 * np.dot(x, b) + r
        value = math.log(s2 + eps) + delta_norm_sq / dim / (s2 + eps)
        grads = 2 / s2 * (x * covs) + 2 / dim / (s2 ** 2 + eps) * ((C @ x - b) * s2 - delta_norm_sq * x * covs)
        return value, grads
    x0 = np.ones(dim)
    if positive:
        eye = np.eye(dim)
        cons = {
            "type": "ineq",
            "fun": lambda x: x,
            "jac": lambda x: eye
        }
        opt = {
            "disp": False
        }
        weights = minimize(func, x0, jac=True, constraints=cons,
                           method="SLSQP", options=opt)["x"]
        weights = np.clip(weights, a_min=0, a_max=None)
    else:
        weights = minimize(func, x0, jac=True)["x"]
    return torch.from_numpy(weights).to(device)


def closed_form(all_grads, target, parametrization, normalization, eps):
    n_weights = len(all_grads)
    with torch.autocast("cuda", enabled=False):
        if parametrization == "abs":
            cov = all_grads @ all_grads.T  # (W, W).
            product = all_grads @ target
            actual_weights = quadratic_program_positive(cov, -product)
            if normalization == "sum":
                scale = n_weights / (actual_weights.sum() + eps)
            elif normalization == "norm":
                scale = math.sqrt(n_weights) / (torch.linalg.norm(actual_weights) + eps)
            else:
                assert normalization == "none"
                scale = 1
            actual_weights = scale * actual_weights
            free_weight = scale
        else:
            assert parametrization == "linear"
            if normalization == "none":
                cov = all_grads @ all_grads.T  # (W, W).
                reg = torch.eye(n_weights, device=cov.device, dtype=cov.dtype)
                product = all_grads @ target
                actual_weights = torch.linalg.solve(cov + eps * reg, product)
            elif normalization == "norm":
                actual_weights = find_closest_unit_norm(all_grads, target)
            else:
                raise NotImplementedError(f"{normalization} normalization in the closed-form algorithm")
            free_weight = 1
    return actual_weights, free_weight


def closed_form_ce(all_grads, target, parametrization, normalization, eps, grad_covs=None):
    if isinstance(grad_covs, Number):
        grad_covs = torch.full([len(all_grads)], grad_covs, dtype=all_grads.dtype, device=all_grads.device)
    if grad_covs is None:
        grad_covs = torch.ones(len(all_grads), dtype=all_grads.dtype, device=all_grads.device)
    cov = all_grads @ all_grads.T  # (W, W).
    product = all_grads @ target
    target_norm_sq = target.T @ target
    # covs: (N).
    n_weights = len(all_grads)
    with torch.autocast("cuda", enabled=False):
        if parametrization == "abs":
            solver = partial(normal_ce_solve, positive=True)
        elif parametrization == "linear":
            solver = normal_ce_solve
        else:
            raise NotImplementedError(f"Unsupported parametrization: {parametrization}.")
        actual_weights = solver(cov, product, target_norm_sq, grad_covs)
        if normalization == "sum":
            scale = n_weights / (actual_weights.sum() + eps)
        elif normalization == "norm":
            scale = math.sqrt(n_weights) / (torch.linalg.norm(actual_weights) + eps)
        else:
            assert normalization == "none"
            scale = 1
        actual_weights = scale * actual_weights
        free_weight = scale
    return actual_weights, free_weight


class CorrHPOptimizer(torch.optim.Optimizer):
    """Correlated Hyperparameter Optimizer.

    Args:
        params: Model parameters (except loss weights). The first params group is for loss weights.
        base_optimizer_cls: The optimizer to use.
        downstream_weight: The weight of the downstream loss in model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        weights_parametrization: Either "linear", "tanh", "abs", or "sigmoid".
        weights_normalization: Whether to normalize weights by their sum or not ("sum", "norm", or "none").
        algorithm: Either "sgd", "closed-form", "closed-form-fast", "closed-form-ce", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be dictionary with "main" and "downstream" keys
            for the main and downstream losses respectively.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        normalize_down_grad: Estimate only the dirrection of the downstream loss gradient.
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
                 weights_parametrization="abs", weights_normalization="norm",
                 algorithm="sgd", ema=0,
                 apply_optimizer_correction=False, normalize_down_grad=True,
                 clip_hp_grad=None, eps=1e-6, **kwargs):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict):
            raise ValueError("Expected at least two param groups with the first group being loss weights.")
        if algorithm not in {"sgd", "closed-form", "closed-form-fast", "closed-form-ce", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if (algorithm in {"closed-form", "closed-form-fast", "closed-form-ce"}) and (weights_parametrization != "linear"):
            if not (algorithm in {"closed-form", "closed-form-ce"} and weights_parametrization == "abs"):
                raise ValueError("Closed-form approach is compatible with linear parametrization only.")
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
        try:
            momentum = dict(ema)
            if set(momentum.keys()) != {"main", "downstream"}:
                raise ValueError(f"Expected dictionary with 'main' and 'downstream' keys.")
            self.main_momentum = momentum["main"]
            self.downstream_momentum = momentum["downstream"]
        except TypeError:
            momentum = float(ema)
            self.main_momentum = momentum
            self.downstream_momentum = momentum
        self.apply_optimizer_correction = apply_optimizer_correction
        self.normalize_down_grad = normalize_down_grad
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps

        # todo: use optimizer state for gradient caches.
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None, HPO_STAGE_FREE: None} | {i: None for i in range(self.n_weights)}

    def step(self, closure, inner=False):
        if not inner:
            raise valueerror("please, use 'hpo_step' function.")
        return self.base_optimizer.step(closure)

    def _update_grads_cache(self, grads, stage=None):
        assert stage in self._grads_cache
        momentum = self.downstream_momentum if stage == HPO_STAGE_DOWNSTREAM else self.main_momentum
        if (self._grads_cache[stage] is not None) and momentum:
            assert len(self._grads_cache[stage]) == len(grads)
            self._grads_cache[stage] = self._grads_cache[stage] * momentum + grads * (1 - momentum)
        else:
            self._grads_cache[stage] = grads
        return self._grads_cache[stage]

    def cache_downstream(self, closure=None):
        """cache downstream gradient for future computations.

        use this method to tune hyperparamers on validation batches.
        """
        assert closure is not None, "need closure"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        downstream_weight = 1
        free_weight = 0
        loss_weights = [0] * self.n_weights
        closure(downstream_weight, free_weight, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
        self._update_grads_cache(self._gather_grads(normalize=self.normalize_down_grad),
                                 stage=HPO_STAGE_DOWNSTREAM)

    def remove_cache(self, stage=None):
        if stage is None:
            self._grads_cache = {name: None for name in self._grads_cache}
        else:
            self._grads_cache[stage] = None

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

        The closure is used like this: closure(target_loss_weight, free_term_weight, *loss_weights, stage=None).
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
                free_weight = 0
                closure(downstream_weight, free_weight, *loss_weights, stage=HPO_STAGE_DOWNSTREAM)
                down_train_grads = self._gather_grads(normalize=self.normalize_down_grad)
                if (not use_cached_downstream) and self.downstream_momentum:
                    self._update_grads_cache(down_train_grads, stage=HPO_STAGE_DOWNSTREAM)
            if use_cached_downstream or self.downstream_momentum:
                down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            else:
                down_grads = down_train_grads
            assert down_grads is not None

            if self.algorithm in {"closed-form", "closed-form-ce", "none"}:
                downstream_weight = 0
                free_weight = 1
                closure(downstream_weight, free_weight, *loss_weights, stage=HPO_STAGE_FREE)
                free_grads = self._gather_grads(apply_optimizer_correction=self.apply_optimizer_correction)
                if self.main_momentum:
                    free_grads = self._update_grads_cache(free_grads, stage=HPO_STAGE_FREE)

            # Caches for normalization differentiation.
            compute_products = self.algorithm in {"sgd", "closed-form-fast"}
            if compute_products:
                products = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            compute_grad_sum = (self.algorithm == "sgd") and (self.weights_normalization in {"sum", "norm"})
            if compute_grad_sum:
                grad_sum = torch.zeros_like(down_grads)
            compute_norms = self.algorithm == "closed-form-fast"
            if compute_norms:
                norms = torch.zeros_like(weights)
            store_all_grads = self.algorithm in {"closed-form", "closed-form-ce"}
            if store_all_grads:
                all_grads = []

            # Compute weights grads.
            downstream_weight = 0
            free_weight = 0
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                closure(downstream_weight, free_weight, *loss_weights, stage=i)
                loss_weights[i] = 0
                loss_grads = self._gather_grads(apply_optimizer_correction=self.apply_optimizer_correction)
                if self.main_momentum:
                    loss_grads = self._update_grads_cache(loss_grads, stage=i)
                if compute_products:
                    products[i] = down_grads @ loss_grads
                if compute_grad_sum:
                    grad_sum += loss_grads * w
                if compute_norms:
                    norms[i] = torch.linalg.norm(loss_grads) + self.eps
                if store_all_grads:
                    all_grads.append(loss_grads)

            free_weight = 1
            if self.algorithm == "closed-form":
                all_grads = torch.stack(all_grads, 0)  # (W, P).
                main_grads_norm = (torch.linalg.norm(all_grads).square() + torch.linalg.norm(free_grads).square()).sqrt() + self.eps
                all_grads /= main_grads_norm
                target = down_grads - free_grads / main_grads_norm
                actual_weights, free_weight = closed_form(all_grads, target,
                                                          parametrization=self.weights_parametrization,
                                                          normalization=self.weights_normalization,
                                                          eps=self.eps)
            elif self.algorithm == "closed-form-ce":
                all_grads = torch.stack(all_grads, 0)  # (W, P).
                main_grads_norm = (torch.linalg.norm(all_grads).square() + torch.linalg.norm(free_grads).square()).sqrt() + self.eps
                all_grads /= main_grads_norm
                target = down_grads - free_grads / main_grads_norm
                actual_weights, free_weight = closed_form_ce(all_grads, target,
                                                             parametrization=self.weights_parametrization,
                                                             normalization=self.weights_normalization,
                                                             eps=self.eps)
            elif self.algorithm == "closed-form-fast":
                assert self.weights_parametrization == "linear"
                actual_weights = products / norms.square()
                scale, norm = self._get_scale_norm(actual_weights)
                actual_weights *= scale / norm
            elif self.algorithm == "sgd":
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
                        weight_grads *= self.clip_hp_grad / grad_norm
            else:
                assert self.algorithm == "none"
                scale, norm = self._get_scale_norm(weights)
                actual_weights = scale * weights / norm

            # Compute model grads.
            downstream_weight = self.downstream_weight
            closure(self.downstream_weight, free_weight, *actual_weights, stage=HPO_STAGE_FINAL)

            if self.algorithm in {"closed-form", "closed-form-ce", "closed-form-fast"}:
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
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        p = self.param_groups[0]["params"][0]
        self._grads_cache.update({k: v.to(device=p.device, dtype=p.dtype) for k, v in state_dict.get("grads_cache", {}).items()})

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
