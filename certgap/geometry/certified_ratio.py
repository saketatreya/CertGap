"""certified_ratio.py — Solves for policy update directions and computes rho_tilde ratios.

Computes:
  - Gradient g^f = E_s [ sum_a nabla_theta pi(a|s) f(s,a) ]
  - Directions v_TV = M^-1 g^f, v_TRPO = F^-1 g^f
  - TV Step Size coefficient: T(v) = E_s [ || J(s) v ||_1 ]  (First-order TV change)
  - Advancement: G(v) = <g^f, v>
  - rho_tilde_coeff = G(v) / T(v)   (Units of Advancement per Unit of TV-risk)

The actual rho_tilde = [2(1-gamma) / Delta_hat] * rho_tilde_coeff.
The ratio rho_tilde_TV / rho_tilde_TRPO is independent of Delta_hat and (1-gamma).
"""
from __future__ import annotations

import torch


def compute_gradient(
    jac: torch.Tensor,
    f_sa: torch.Tensor,
) -> torch.Tensor:
    """Compute the policy gradient g^f.

    Args:
        jac: Jacobian (N, A, P).
        f_sa: Action values (N, A).

    Returns:
        Gradient scaled by 1/N, shape (P,).
    """
    # g = (1/N) * sum_i sum_a J_ia * f_ia
    # jac is (N, A, P), f is (N, A)
    g = torch.einsum("nap,na->p", jac, f_sa) / jac.shape[0]
    return g


def compute_tv_step_size(
    jac: torch.Tensor,
    v: torch.Tensor,
) -> float:
    """Compute the expected L1 change T(v) = E_s [ || J(s) v ||_1 ]."""
    # J(s)v is (N, A)
    delta_pi = torch.einsum("nap,p->na", jac, v)
    l1_change = torch.norm(delta_pi, p=1, dim=1).mean().item()
    return l1_change


def solve_for_direction(
    matrix: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Solve H v = g."""
    # matrix is (P, P), gradient is (P,)
    # Use lstsq or solve for efficiency.
    # Regularization is already in the matrix from compute_fishers.
    try:
        v = torch.linalg.solve(matrix, gradient)
    except torch.linalg.LinAlgError:
        # Fallback to lstsq if still singular
        v, _, _, _ = torch.linalg.lstsq(matrix, gradient.unsqueeze(1))
        v = v.squeeze(1)
    return v


def compute_certified_stats(
    fisher_dict: dict,
    f_sa: torch.Tensor,
) -> dict:
    """Compute Advancement, Risk, and rho_tilde coefficients across dual spectral floors.
    
    Args:
        fisher_dict: Output from compute_fishers (containing F_train, M_train, F_diag, M_diag, jac).
        f_sa: Action values (N, A).
    """
    jac = fisher_dict["jac"]
    g = compute_gradient(jac, f_sa)
    
    # We use M_diag and F_diag for the "True" theoretical Signal
    # and M_train, F_train for the "Numerical Stability" metric.
    
    # 1. Directions derived from Diagnostic Matrix (High precision)
    v_tv = solve_for_direction(fisher_dict["M_diag"], g)
    v_kl = solve_for_direction(fisher_dict["F_diag"], g)
    
    # TV Direction stats
    g_tv = torch.dot(g, v_tv).item()
    t_tv = compute_tv_step_size(jac, v_tv)
    coeff_tv = g_tv / (t_tv + 1e-10)
    
    # KL Direction stats (TRPO)
    g_kl = torch.dot(g, v_kl).item()
    t_kl = compute_tv_step_size(jac, v_kl)
    coeff_kl = g_kl / (t_kl + 1e-10)
    
    # 2. Condition numbers for both DIAGONSTIC and TRAINING regimes
    with torch.no_grad():
        w_f_diag = torch.linalg.eigvalsh(fisher_dict["F_diag"])
        w_m_diag = torch.linalg.eigvalsh(fisher_dict["M_diag"])
        w_f_train = torch.linalg.eigvalsh(fisher_dict["F_train"])
        w_m_train = torch.linalg.eigvalsh(fisher_dict["M_train"])
        
        # Diagnostic conditioning (True Signal)
        kappa_f_diag = (w_f_diag[-1] / torch.clamp(w_f_diag[0], min=1e-15)).item()
        kappa_m_diag = (w_m_diag[-1] / torch.clamp(w_m_diag[0], min=1e-15)).item()
        min_f_diag = w_f_diag[0].item()
        min_m_diag = w_m_diag[0].item()
        
        # Training conditioning (Numerical Stability)
        kappa_f_train = (w_f_train[-1] / torch.clamp(w_f_train[0], min=1e-15)).item()
        kappa_m_train = (w_m_train[-1] / torch.clamp(w_m_train[0], min=1e-15)).item()


    return {
        "g_tv": g_tv,
        "t_tv": t_tv,
        "coeff_tv": coeff_tv,
        "g_kl": g_kl,
        "t_kl": t_kl,
        "coeff_kl": coeff_kl,
        "ratio": torch.clamp(torch.tensor(coeff_tv / (coeff_kl + 1e-10)), -50, 50).item(),
        "kappa_f_diag": kappa_f_diag,

        "kappa_m_diag": kappa_m_diag,
        "min_f_diag": min_f_diag,
        "min_m_diag": min_m_diag,
        "kappa_f_train": kappa_f_train,
        "kappa_m_train": kappa_m_train,
    }
