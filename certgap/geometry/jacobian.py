"""Tools for computing Policy Jacobian and Fisher matrices (KL and TV-Euclidean).

The geometry of the policy space is determined by the metric on the simplex.
- KL Geometry: F(theta) = E [ J^T diag(pi)^-1 J ]
- TV Geometry: M(theta) = E [ J^T J ]  (Euclidean metric on probabilities)

Where J is the Jacobian of the policy probabilities w.r.t the parameters theta.
"""
from __future__ import annotations

import torch
from torch import nn


def get_params_and_names(model: nn.Module) -> tuple[list[torch.Tensor], list[str]]:
    params = []
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(name)
    return params, names


def compute_policy_jacobian(
    policy_net: nn.Module,
    states: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jacobian of policy probabilities w.r.t theta.

    Args:
        policy_net: Policy neural network.
        states: Tensor of batch shape (N, state_dim).

    Returns:
        Jacobian J of shape (N, action_dim, num_params).
    """
    # Fix the state and treat params as input for jacobian
    params, _ = get_params_and_names(policy_net)
    num_params = sum(p.numel() for p in params)
    N = states.shape[0]
    
    # Extract flattened params for functional calls
    curr_params_vec = nn.utils.parameters_to_vector(params)

    def func_pi(params_vec: torch.Tensor) -> torch.Tensor:
        # Load params into a temporary version of the network
        # We need to be careful with functional parameterization
        # A simpler way is to use torch.func.functional_call if available
        # or manual reconstruction.
        
        # Using a closure and manual assignment is tricky with autograd.functional.
        # Let's use the standard way: functional_call.
        from torch.func import functional_call
        
        # Reconstruct the parameter dict
        param_dict = {}
        pointer = 0
        for name, p in policy_net.named_parameters():
            numel = p.numel()
            param_dict[name] = params_vec[pointer:pointer+numel].view_as(p)
            pointer += numel
        
        # Forward pass (probabilities)
        logits = functional_call(policy_net, param_dict, states)
        probs = torch.softmax(logits, dim=-1)
        return probs

    # Jac shape will be (N, A, num_params)
    # autograd.functional.jacobian expects a flat input if we want a flat jacobian w.r.t it
    jac = torch.autograd.functional.jacobian(func_pi, curr_params_vec)
    return jac


def compute_fishers(
    policy_net: nn.Module,
    states: torch.Tensor,
    reg_train: float = 1e-4,
    reg_diag: float = 1e-9,
) -> dict:
    """Compute KL-Fisher F(theta) and TV-Fisher M(theta) with dual regularization.

    Returns:
        dict with:
          - F_train: (P, P) with reg_train
          - M_train: (P, P) with reg_train
          - F_diag: (P, P) with reg_diag
          - M_diag: (P, P) with reg_diag
          - pi: (N, A)
          - jac: (N, A, P)
    """
    # 1. Get probabilities
    with torch.no_grad():
        logits = policy_net(states)
        pi = torch.softmax(logits, dim=-1)
    
    # 2. Get Jacobian (N, A, P)
    jac = compute_policy_jacobian(policy_net, states)
    N, A, P = jac.shape
    
    # 3. Base Matrices (unregularized)
    jac_flat = jac.view(N * A, P)
    M_base = (jac_flat.T @ jac_flat) / N
    
    inv_pi = 1.0 / (pi + 1e-10)
    weights = torch.sqrt(inv_pi) / (N ** 0.5)
    weighted_jac = jac * weights[:, :, None]
    weighted_jac_flat = weighted_jac.view(N * A, P)
    F_base = weighted_jac_flat.T @ weighted_jac_flat
    
    # 4. Add regularization
    eye = torch.eye(P, device=jac.device)
    
    return {
        "F_train": F_base + reg_train * eye,
        "M_train": M_base + reg_train * eye,
        "F_diag": F_base + reg_diag * eye,
        "M_diag": M_base + reg_diag * eye,
        "pi": pi,
        "jac": jac
    }

