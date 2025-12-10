import math
import torch
import numpy as np
from functools import partial
from numbers import Number

from scipy.optimize import bisect, minimize, linprog


def _zero_last(x):
    x[-1] = 0
    return x


def _find_lambda_closest_unit_norm_np(cov, b, eps=1e-6, steps=100):
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


class ProcrustesSolver:
    """Senushkin D. et al. Independent component alignment for multi-task learning // CVPR 2023."""
    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def apply(grads, unit_scale=False):

        assert len(grads.shape) == 3, \
            f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():

            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e, UPLO="U")
            tol = torch.max(singulars) * max(cov_grad_matrix_e.shape[-2:]) * torch.finfo().eps
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if unit_scale:
                weights = basis
            else:
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            weights = weights / torch.sqrt(singulars).view(1, -1)
            # weights = weights / singulars.view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars


def solve_lstsq_unit_norm(basis, target, eps=1e-6, steps=100):
    """Solve Ax=B with |x| = 1."""
    # Handle a special case with zero basis.
    zero_mask = torch.linalg.norm(basis, dim=1) < eps  # (W).
    zero_index = zero_mask.nonzero()[0, 0].item() if zero_mask.any() else None

    cov = (basis @ basis.T).cpu().double().numpy()  # (W, W).
    b = (basis @ target).cpu().double().numpy()  # (W).
    l = _find_lambda_closest_unit_norm_np(cov, b)
    reg = np.eye(len(cov))
    weights = np.linalg.lstsq(cov + l * reg, b)[0]

    if zero_index is not None:
        assert abs(weights[zero_index]) < eps
        weights[zero_index] = np.sqrt(max(0, 1 - np.sum(weights ** 2)))
    return torch.from_numpy(weights).to(device=basis.device)


def solve_qp(C, b, steps=100, method="SLSQP", eps=1e-6,
             positive=False, norm=False):
    """Minimize 0.5 x^T C x + b^T x.

    Args:
      positive: Whether to add x > 0 constraint or not.
      norm: Whether to add |x| = 1 constraint or not. Boolean or a number of features to normalize.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()
    func = lambda x: 0.5 * x.T @ C @ x + b @ x
    jac = lambda x: C @ x + b
    cons = []
    if positive:
        eye = np.eye(n)
        cons.append({
            "type": "ineq",
            "fun": lambda x: x,
            "jac": lambda x: eye
        })
    if norm:
        if isinstance(norm, bool):
            norm_size = n
        else:
            norm_size = int(norm)
        cons.append({
            "type": "eq",
            "fun": lambda x: (x[:norm_size] ** 2).sum() - 1,
            "jac": lambda x: 2 * np.concatenate([x[:norm_size], np.zeros_like(x[norm_size:])])
        })
    opt = {
        "disp": False,
        "maxiter": steps
    }
    x0 = np.ones(n) / n
    weights = minimize(func, x0, jac=jac, constraints=cons,
                       method=method, options=opt)["x"]
    if positive:
        weights = np.clip(weights, a_min=0, a_max=None)
    if norm:
        scale = np.linalg.norm(weights[:norm_size])
        if scale < eps:
            weights = np.ones_like(weights)
            weights = np.concatenate([np.ones_like(weights[:norm_size]), weights[norm_size:]])
            scale = np.linalg.norm(weights[:norm_size])
        weights = np.concatenate([weights[:norm_size] / (scale + eps), weights[norm_size:]])
    return torch.from_numpy(weights).to(dtype=dtype, device=device)


def solve_ce(C, b, r, covs, cov_err, dim, log_scale=1, scaled=False, steps=100, method="SLSQP", eps=1e-6,
             positive=False, norm=False):
    """Minimize d log s^2 + (x^T C x + 2 * b^T x + r) / s^2, where s^2 = sum w^2_i covs_i + cov_err.

    Args:
      positive: Whether to add x > 0 constraint or not.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()
    r = r.item()
    covs = covs.cpu().double().numpy()
    cov_err = cov_err.cpu().double().numpy()

    def func(x):
        s2 = ((x ** 2) * covs).sum() + cov_err + eps
        delta_norm_sq = x.T @ C @ x + 2 * np.dot(x, b) + r
        value = dim * log_scale * math.log(s2) + delta_norm_sq / s2
        grads = 2 * dim * log_scale / s2 * (x * covs) + 2 / (s2 ** 2 + eps) * ((C @ x + b) * s2 - delta_norm_sq * x * covs)
        if scaled:
            value -= dim * math.log(x[-1] ** 2 + eps)
            lx = x[-1]
            if abs(lx) < eps:
                if lx > 0:
                    lx += eps
                else:
                    lx -= eps
            grads[-1] -= 2 * dim / lx
        return value, grads
    x0 = np.ones(n) / n

    cons = []
    if positive:
        eye = np.eye(n)
        cons.append({
            "type": "ineq",
            "fun": lambda x: x,
            "jac": lambda x: eye
        })
    elif scaled:
        eye_last = np.zeros(n)
        eye_last[-1] = 1
        cons.append({
            "type": "ineq",
            "fun": lambda x: x[-1],
            "jac": lambda x: eye_last
        })

    if norm:
        size = n - 1 if scaled else n
        cons.append({
            "type": "eq",
            "fun": lambda x: (x[:size] ** 2).sum() - 1,
            "jac": lambda x: _zero_last(2 * x)
        })

    opt = {
        "disp": False,
        "maxiter": steps
    }

    weights = minimize(func, x0, jac=True, constraints=cons or None,
                       method="SLSQP", options=opt)["x"]
    if positive:
        weights = np.clip(weights, a_min=0, a_max=None)
    elif scaled:
        weights[-1] = max(0, weights[-1])
    return torch.from_numpy(weights).to(dtype=dtype, device=device)


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form(basis, target, covs=None, positive=False, norm=False, scale_target=False, eps=1e-6):
    """"Find a combination of basis vectors that is close to target.

    Args:
        basis: Basis vectors with shape (W, P).
        target: Target vector with shape (P).
        covs: Scale vector with shape (P) or a number.
        positive: Find positive weights.

    Returns:
        Weights with shape (W).
    """
    if covs is None:
        covs = 1
    elif (not isinstance(covs, Number)) and (covs.shape != target.shape):
        raise ValueError("Covariances must be number or vector.")
    if isinstance(covs, Number):
        scales = 1  # Scale doesn't affect the result.
        scaled_basis = basis
        scaled_target = target
    else:
        scales = covs.sqrt()
        mean_scale = scales.mean().item()
        if mean_scale < eps:
            scales = 1
        else:
            scales /= (mean_scale + eps)
        scaled_basis = basis / scales
        scaled_target = target / scales
    if scale_target:
        n_weights = len(basis)
        prods = -(scaled_basis @ scaled_target)
        C = torch.empty((n_weights + 1, n_weights + 1), dtype=basis.dtype, device=basis.device)
        C[:n_weights, :n_weights] = scaled_basis @ scaled_basis.T  # (W, W).
        C[:n_weights, n_weights] = prods
        C[n_weights, :n_weights] = prods
        C[n_weights, n_weights] = target @ target
        b = torch.zeros_like(C[0])
    else:
        C = scaled_basis @ scaled_basis.T  # (W, W).
        b = -(scaled_basis @ scaled_target)  # (W).

    scale = C.mean() + eps
    C /= scale
    b /= scale

    norm_size = len(basis) if norm else False
    weights = solve_qp(C, b, eps=eps, positive=positive, norm=norm_size)
    if scale_target:
        weights = weights[:-1]
    return weights


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form_spherical(basis, target, covs=None, cov_err=0, positive=False, norm=False, scale_target=False, eps=1e-6):
    """"Find a combination of basis vectors that is close to target.

    NOTE: This solver works only with spherical distributions.

    Args:
        basis: Basis vectors with shape (W, P).
        target: Target vector with shape (P).
        covs: Scale vector with shape (W) or a number.
        cov_err: An additional covariance of the error.
        positive: Find positive weights.

    Returns:
        Weights with shape (W).
    """
    if covs is None:
        covs = 1
    elif (not isinstance(covs, Number)) and (covs.shape != (len(basis),)):
        raise ValueError("Covariances must be number or vector.")
    if isinstance(covs, Number):
        covs = torch.full_like(basis[:, 0], covs)
    if isinstance(cov_err, Number):
        cov_err = torch.full_like(target[0], cov_err)
    if scale_target:
        n_weights = len(basis)
        prods = -(basis @ target)
        C = torch.empty((n_weights + 1, n_weights + 1), dtype=basis.dtype, device=basis.device)
        C[:n_weights, :n_weights] = basis @ basis.T  # (W, W).
        C[:n_weights, n_weights] = prods
        C[n_weights, :n_weights] = prods
        C[n_weights, n_weights] = target @ target
        b = torch.zeros_like(C[0])
        r = torch.zeros_like(C[0, 0])
        covs = torch.cat([covs, torch.zeros_like(covs[:1])])
    else:
        C = basis @ basis.T  # (W, W).
        b = -(basis @ target)  # (W).
        r = target @ target  # Scalar

    covs = covs.double()
    cov_err = cov_err.double()
    scale = (covs.mean() + cov_err).item()
    if scale < eps ** 2:
        scale = 1
    covs = covs / scale
    cov_err = cov_err / scale
    log_scale = scale

    weights = solve_ce(C, b, r, covs, cov_err, dim=len(target), log_scale=log_scale, scaled=scale_target, eps=eps, positive=positive, norm=norm)
    if scale_target:
        weights = weights[:-1]
    return weights


# TODO: faster, more ITERS.
@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form_fw(basis, target, positive=False, normalization="sum", scale_target=True, eps=1e-6, max_iter=100):
    """"Find a combination of basis vectors that is close to a scaled target.

    Args:
        basis: Basis vectors with shape (W, P).
        target: Target vector with shape (P).
        positive: Find positive weights.
        normalization: Either "sum", "norm", or "none".
        scale_target: Allow scaling of a target vector.

    Returns:
        Weights with shape (W) and an optimal target scale S.
    """
    n_weights = len(basis)

    C = (basis @ basis.T).cpu().numpy()  # (W, W).
    b = (basis @ target).cpu().numpy()  # (W).
    r = (target @ target).item()

    if scale_target:
        cov = np.empty((n_weights + 1, n_weights + 1), dtype=np.double)
        cov[:n_weights, :n_weights] = C
        cov[:n_weights, n_weights] = -b
        cov[n_weights, :n_weights] = -b
        cov[n_weights, n_weights] = r
    else:
        cov = C

    max_scale = 10 * torch.linalg.norm(basis, dim=1).max().item() / (torch.linalg.norm(target).item() + eps)
    extra_param_num = int(bool(scale_target))

    if normalization == "none":
        A_eq = None
        b_eq = None
        norms = torch.linalg.norm(basis, dim=1).cpu().numpy()  # (W).
        max_weights = np.clip(np.abs(b) / (norms ** 2 + eps), 1, None)
    elif normalization == "sum":
        A_eq = np.ones((1, n_weights + extra_param_num))
        if scale_target:
            A_eq[0, -1] = 0
        b_eq = np.ones(1)
        max_weights = [None] * n_weights
    else:
        raise NotImplementedError(f"Normalization: {normalization}")


    if positive:
        bounds = [(0, w) for w in max_weights]
    else:
        bounds = [(-w, w) for w in max_weights]
    if scale_target:
        bounds = bounds + [(0, max_scale)]

    # Apply Frank-Wolfe optimization.
    x = np.full((n_weights + extra_param_num), 1 / n_weights)
    if scale_target:
        x[-1] = 1  # Target scale.

    for k in range(max_iter):
        # Step 1.
        if scale_target:
            grad = cov @ x
        else:
            grad = cov @ x - b
        r = linprog(grad, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        if not r.success:
            raise RuntimeError(f"Linprog failed: {r.status}")
        s = r.x

        # Step 2.
        delta = s - x
        delta_c = (delta[None] @ cov)[0]  # (W).
        ca = delta_c @ delta  # Non-negative.
        if scale_target:
            cb = delta_c @ x
        else:
            cb = delta_c @ x - b @ delta
        alpha = - 2 * cb / (ca + eps)
        alpha = min(alpha, 1)
        alpha = max(alpha, 0)

        # Step 3.
        x = x + alpha * (s - x)

    if scale_target:
        return torch.from_numpy(x[:-1]).to(device=target.device, dtype=target.dtype), float(x[-1])
    else:
        return torch.from_numpy(x).to(device=target.device, dtype=target.dtype), 1


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form_trmse(basis, target, covs=None, positive=False, eps=1e-6):
    """"Find a combination of basis vectors that is close to target.

    Args:
        basis: Basis vectors with shape (W, P).
        target: Target vector with shape (P).
        covs: Scale vector with shape (W, P) or a number for basis vectors.
        positive: Find positive weights.

    Returns:
        Weights with shape (W).
    """
    if (covs is not None) and (covs.shape != basis.shape):
        raise ValueError("Covariances must be a matrix with shape (W, P).")
    C = basis @ basis.T  # (W, W).
    b = -(basis @ target)  # (W).

    if covs is not None:
        C = C + 2 * torch.diag(covs.sum(1))

    weights = solve_qp(C, b, eps=eps, positive=positive)
    if positive and (not weights.isfinite().all()):
        weights = solve_qp(C, b, eps=eps, positive=False)
        weights = weights.clip(min=0)
    return weights
