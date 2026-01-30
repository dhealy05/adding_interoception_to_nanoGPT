"""Apply regime-based interventions to training data and state."""

from typing import Iterable, Tuple

import torch

from regimes import Regime


STATS_FIELDS = ["loss", "grad_norm", "step_frac", "entropy", "top1_conf"]


def apply_input_mask(X: torch.Tensor, prob: float, mask_id: int) -> torch.Tensor:
    if prob <= 0.0:
        return X
    mask = torch.rand(X.shape, device=X.device) < prob
    masked = X.clone()
    masked[mask] = mask_id
    return masked


def apply_input_random(X: torch.Tensor, prob: float, vocab_size: int) -> torch.Tensor:
    if prob <= 0.0:
        return X
    mask = torch.rand(X.shape, device=X.device) < prob
    random_tokens = torch.randint_like(X, 0, vocab_size)
    return torch.where(mask, random_tokens, X)


def apply_target_patch_random(Y: torch.Tensor, prob: float, vocab_size: int) -> torch.Tensor:
    if prob <= 0.0:
        return Y
    mask = torch.rand(Y.shape, device=Y.device) < prob
    random_targets = torch.randint_like(Y, 0, vocab_size)
    return torch.where(mask, random_targets, Y)


def _bias_vector_from_params(params, stats_dim: int, device: torch.device) -> torch.Tensor:
    if "bias" in params:
        bias = params["bias"]
        if isinstance(bias, (list, tuple)) and len(bias) == stats_dim:
            return torch.tensor(bias, device=device, dtype=torch.float32)
        raise ValueError("stats bias must be a list with length stats_dim")
    bias_vec = torch.zeros(stats_dim, device=device, dtype=torch.float32)
    for idx, name in enumerate(STATS_FIELDS):
        key = f"{name}_bias"
        if key in params:
            bias_vec[idx] = float(params[key])
    return bias_vec


def apply_stats_bias(stats_prev: torch.Tensor, intensity: float, params: dict) -> torch.Tensor:
    bias_vec = _bias_vector_from_params(params, stats_prev.numel(), stats_prev.device)
    return stats_prev + intensity * bias_vec


def apply_stats_noise(stats_prev: torch.Tensor, intensity: float, params: dict) -> torch.Tensor:
    std = float(params.get("std", 1.0))
    return stats_prev + torch.randn_like(stats_prev) * (std * intensity)


def apply_self_clamp(self_state: torch.Tensor, params: dict) -> torch.Tensor:
    min_val = float(params.get("min", -1.0))
    max_val = float(params.get("max", 1.0))
    return torch.clamp(self_state, min=min_val, max=max_val)


def apply_self_noise(self_state: torch.Tensor, intensity: float, params: dict) -> torch.Tensor:
    std = float(params.get("std", 0.01))
    return self_state + torch.randn_like(self_state) * (std * intensity)


def apply_self_scale(self_state: torch.Tensor, intensity: float, params: dict) -> torch.Tensor:
    scale = float(params.get("scale", 1.0))
    return self_state * (scale * intensity)


def apply_interventions(
    X: torch.Tensor,
    Y: torch.Tensor,
    regimes: Iterable[Regime],
    model_config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_out, Y_out = X, Y
    for regime in regimes:
        if "input" in regime.channels:
            mode = regime.params.get("mode", "mask_tokens")
            if mode == "mask_tokens":
                mask_id = int(regime.params.get("mask_id", 0))
                X_out = apply_input_mask(X_out, regime.intensity, mask_id)
            elif mode == "random_tokens":
                X_out = apply_input_random(X_out, regime.intensity, model_config.vocab_size)
            else:
                raise ValueError(f"unknown input mode: {mode}")
        if "target" in regime.channels:
            mode = regime.params.get("mode", "random_targets")
            if mode == "random_targets":
                Y_out = apply_target_patch_random(Y_out, regime.intensity, model_config.vocab_size)
            else:
                raise ValueError(f"unknown target mode: {mode}")
    return X_out, Y_out


def apply_stats_interventions(stats_prev: torch.Tensor, regimes: Iterable[Regime]) -> torch.Tensor:
    out = stats_prev
    for regime in regimes:
        if "stats" not in regime.channels:
            continue
        mode = regime.params.get("mode", "bias")
        if mode == "bias":
            out = apply_stats_bias(out, regime.intensity, regime.params)
        elif mode == "noise":
            out = apply_stats_noise(out, regime.intensity, regime.params)
        else:
            raise ValueError(f"unknown stats mode: {mode}")
    return out


def apply_self_interventions(self_state: torch.Tensor, regimes: Iterable[Regime]) -> torch.Tensor:
    out = self_state
    for regime in regimes:
        if "self" not in regime.channels:
            continue
        mode = regime.params.get("mode", "clamp")
        if mode == "clamp":
            out = apply_self_clamp(out, regime.params)
        elif mode == "noise":
            out = apply_self_noise(out, regime.intensity, regime.params)
        elif mode == "scale":
            out = apply_self_scale(out, regime.intensity, regime.params)
        else:
            raise ValueError(f"unknown self mode: {mode}")
    return out


def compute_lr_multiplier(regimes: Iterable[Regime]) -> float:
    mult = 1.0
    for regime in regimes:
        if "optim" not in regime.channels:
            continue
        mode = regime.params.get("mode", "lr_mult")
        if mode == "lr_mult":
            factor = float(regime.params.get("mult", regime.intensity))
            mult *= factor
        else:
            raise ValueError(f"unknown optim mode: {mode}")
    return mult
