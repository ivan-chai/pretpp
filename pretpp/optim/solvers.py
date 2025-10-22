import math
import torch
import numpy as np
from functools import partial
from numbers import Number

from scipy.optimize import bisect, minimize


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
      norm: Whether to add |x| = 1 constraint or not.
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
        cons.append({
            "type": "eq",
            "fun": lambda x: (x ** 2).sum() - 1,
            "jac": lambda x: 2 * x
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
        scale = np.linalg.norm(weights)
        if scale < eps:
            weights = np.ones_like(weights)
            scale = np.linalg.norm(weights)
        weights = weights / (scale + eps)
    return torch.from_numpy(weights).to(dtype=dtype, device=device)


def solve_ce(C, b, r, covs, cov_err, dim, log_scale=1, steps=100, method="SLSQP", eps=1e-6,
             positive=False):
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

    opt = {
        "disp": False,
        "maxiter": steps
    }

    weights = minimize(func, x0, jac=True, constraints=cons or None,
                       method="SLSQP", options=opt)["x"]
    if positive:
        weights = np.clip(weights, a_min=0, a_max=None)
    return torch.from_numpy(weights).to(dtype=dtype, device=device)


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form_unit_norm(basis, target, covs=None, positive=False, eps=1e-6):
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
    else:
        scales = covs.sqrt()
        mean_scale = scales.mean().item()
        if mean_scale < eps:
            scales = 1
        else:
            scales /= (mean_scale + eps)
    scaled_basis = basis / scales
    scaled_target = target / scales
    C = scaled_basis @ scaled_basis.T  # (W, W).
    b = -(scaled_basis @ scaled_target)  # (W).

    scale = C.mean() + eps
    C /= scale
    b /= scale

    weights = solve_qp(C, b, eps=eps, positive=positive)
    return weights


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def closed_form_spherical(basis, target, covs=None, cov_err=0, positive=False, eps=1e-6):
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
    C = basis @ basis.T  # (W, W).
    b = -(basis @ target)  # (W).
    r = target @ target  # Scalar

    scale = (covs.mean() + cov_err).item()
    if scale < eps:
        scale = 1
    covs = covs / scale
    cov_err = cov_err / scale
    log_scale = scale

    weights = solve_ce(C, b, r, covs, cov_err, dim=len(target), log_scale=log_scale, eps=eps, positive=positive)
#    if weights[1] > 30 * weights[0]:
#        import ipdb
#        ipdb.set_trace()
    return weights
