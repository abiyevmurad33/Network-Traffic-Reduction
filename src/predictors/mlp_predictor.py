from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.predictors.base import Obs, Predictor


def _torch_load_compat(path: str | Path, device: str) -> Any:
    """
    PyTorch 2.6 changed torch.load default weights_only=True.
    We explicitly try weights_only=False first (safe if your ckpt is yours).
    If the runtime doesn't accept weights_only kwarg, fall back.
    """
    p = str(path)
    try:
        return torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        # Older torch that doesn't have weights_only argument
        return torch.load(p, map_location=device)


def _extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """
    Supports:
      - dict with 'model_state'
      - dict with 'state_dict'
      - raw state_dict (dict of tensors)
    """
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

        # Heuristic: if values look like tensors, treat as state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]

    raise ValueError(
        "Unsupported checkpoint format. Expected dict with 'model_state'/'state_dict' or raw state_dict."
    )


def _infer_linear_shapes_from_state(state: dict[str, torch.Tensor]) -> tuple[int, int, list[int]]:
    """
    Infer MLP dims from Linear weights in state_dict.
    Returns: (in_dim, out_dim, hidden_dims)
    """
    # Collect Linear weights in order-ish. We'll sort keys to be stable.
    weight_items = [(k, v) for k, v in state.items() if k.endswith(".weight") and v.ndim == 2]
    if not weight_items:
        raise ValueError(
            "Could not infer network dims: no 2D '.weight' tensors found in checkpoint state_dict."
        )

    weight_items.sort(key=lambda kv: kv[0])

    # Assume first weight corresponds to first Linear: [hidden, in]
    # last weight corresponds to last Linear: [out, hidden_last]
    first_w = weight_items[0][1]
    last_w = weight_items[-1][1]

    in_dim = int(first_w.shape[1])
    out_dim = int(last_w.shape[0])

    # Hidden dims: take each intermediate layer's out_features (shape[0]),
    # excluding last layer's out_features.
    hidden_dims: list[int] = []
    for _, w in weight_items[:-1]:
        hidden_dims.append(int(w.shape[0]))

    # If we only have two Linear layers, hidden_dims will have exactly one entry.
    # If there's a deeper net, it will have multiple.
    return in_dim, out_dim, hidden_dims


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mask_to_bits(mask: int) -> list[int]:
    """
    Encodes your Zandronum action bitmask into a compact 8-dim multi-hot:
      [ATTACK, USE, JUMP, SHIFT, W, S, A, D]

    Based on your mapping:
      Attack: 1
      Use:    2
      Jump:   4
      Shift:  256
      W:      8192
      S:      4096
      A:      2048
      D:      1024
    """
    return [
        1 if (mask & 1) else 0,  # attack
        1 if (mask & 2) else 0,  # use
        1 if (mask & 4) else 0,  # jump
        1 if (mask & 256) else 0,  # shift
        1 if (mask & 8192) else 0,  # W
        1 if (mask & 4096) else 0,  # S
        1 if (mask & 2048) else 0,  # A
        1 if (mask & 1024) else 0,  # D
    ]


@dataclass
class MLPPredictor(Predictor):
    """
    Simple MLP-based 1-step predictor.

    Input features are built from the last observed state + action encoding.
    Output is interpreted as delta to apply to (x,y,z,yaw,pitch) for next tic.

    This file is designed to be robust to checkpoint format differences:
    - supports ckpt with model_state/state_dict/raw
    - supports missing normalization stats (identity defaults)
    """

    ckpt_path: str
    device: str = "cpu"

    # Model + stats loaded at init
    net: nn.Module = field(init=False)
    in_dim: int = field(init=False)
    out_dim: int = field(init=False)

    x_mean: np.ndarray = field(init=False)
    x_std: np.ndarray = field(init=False)
    y_mean: np.ndarray = field(init=False)
    y_std: np.ndarray = field(init=False)

    _last: Obs | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        ckpt = _torch_load_compat(self.ckpt_path, self.device)
        state = _extract_state_dict(ckpt)

        in_dim, out_dim, hidden_dims = _infer_linear_shapes_from_state(state)
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Build a compatible architecture and load weights
        self.net = _MLP(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim)
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

        # Load normalization stats if present; otherwise identity.
        # We accept lists/np arrays/torch tensors.
        def _arr(key: str, n: int, default: float) -> np.ndarray:
            if isinstance(ckpt, dict) and key in ckpt:
                v = ckpt[key]
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                a = np.array(v, dtype=np.float32).reshape(-1)
                if a.size != n:
                    # If mismatch, fall back to identity rather than silently broadcasting
                    return np.full((n,), default, dtype=np.float32)
                return a
            return np.full((n,), default, dtype=np.float32)

        self.x_mean = _arr("x_mean", self.in_dim, 0.0)
        self.x_std = _arr("x_std", self.in_dim, 1.0)
        self.y_mean = _arr("y_mean", self.out_dim, 0.0)
        self.y_std = _arr("y_std", self.out_dim, 1.0)

        # Avoid divide-by-zero
        self.x_std = np.where(self.x_std == 0.0, 1.0, self.x_std)
        self.y_std = np.where(self.y_std == 0.0, 1.0, self.y_std)

    def reset(self) -> None:
        self._last = None

    def observe(self, obs: Obs) -> None:
        self._last = obs

    def _build_x(self, obs: Obs, action_mask: int) -> np.ndarray:
        # Use float64 internally to avoid overflow, then cast down safely.
        def _finite(v: float) -> float:
            return float(v) if np.isfinite(v) else 0.0

        base64 = np.array(
            [
                _finite(obs.x),
                _finite(obs.y),
                _finite(obs.z),
                _finite(obs.yaw),
                _finite(obs.pitch),
            ],
            dtype=np.float64,
        )

        bits64 = np.array(_mask_to_bits(action_mask), dtype=np.float64)
        mask_scalar64 = np.array([float(action_mask)], dtype=np.float64)

        candidates64 = [
            np.concatenate([base64, bits64], axis=0),  # 13
            np.concatenate([base64, mask_scalar64], axis=0),  # 6
            base64,  # 5
            np.concatenate([base64, bits64, mask_scalar64], axis=0),  # 14
        ]

        for x64 in candidates64:
            if x64.size == self.in_dim:
                return x64.astype(np.float32)

        x64 = candidates64[0]  # most informative
        if x64.size < self.in_dim:
            pad = np.zeros((self.in_dim - x64.size,), dtype=np.float64)
            x64 = np.concatenate([x64, pad], axis=0)
        else:
            x64 = x64[: self.in_dim]

        return x64.astype(np.float32)

    @torch.no_grad()
    def step(self, action_mask: int) -> Obs:
        """
        Predict next tic given action_mask applied at current tic.
        Output is interpreted as delta for [x,y,z,yaw,pitch] by default.
        """
        if self._last is None:
            raise RuntimeError("MLPPredictor.step() called before observe().")

        x = self._build_x(self._last, action_mask)
        x_n = (x - self.x_mean) / self.x_std

        xt = torch.from_numpy(x_n).to(self.device).float().unsqueeze(0)  # [1, in_dim]
        y_n = self.net(xt).squeeze(0).detach().cpu().numpy().astype(np.float32)
        y = (y_n * self.y_std) + self.y_mean

        # If model outputs go non-finite, zero them out to avoid poisoning the rollout.
        if not np.all(np.isfinite(y)):
            y = np.zeros_like(y, dtype=np.float32)

        # Interpret output as per-tic deltas.
        dx = dy = dz = dyaw = dpitch = 0.0
        if y.size >= 5:
            dx, dy, dz, dyaw, dpitch = (
                float(y[0]),
                float(y[1]),
                float(y[2]),
                float(y[3]),
                float(y[4]),
            )

        # Hard clamps (units/tic and deg/tic) to prevent runaway integration.
        # These are conservative; adjust later based on observed max speeds in logs.
        MAX_POS_STEP = 64.0  # map units per tic
        MAX_YAW_STEP = 30.0  # degrees per tic
        MAX_PITCH_STEP = 30.0  # degrees per tic

        dx = float(np.clip(dx, -MAX_POS_STEP, MAX_POS_STEP))
        dy = float(np.clip(dy, -MAX_POS_STEP, MAX_POS_STEP))
        dz = float(np.clip(dz, -MAX_POS_STEP, MAX_POS_STEP))
        dyaw = float(np.clip(dyaw, -MAX_YAW_STEP, MAX_YAW_STEP))
        dpitch = float(np.clip(dpitch, -MAX_PITCH_STEP, MAX_PITCH_STEP))

        yaw_raw = float(self._last.yaw + dyaw)
        yaw_norm = ((yaw_raw + 180.0) % 360.0) - 180.0

        pitch_raw = float(self._last.pitch + dpitch)
        pitch_clamped = float(np.clip(pitch_raw, -89.0, 89.0))

        nxt = Obs(
            tic=int(self._last.tic + 1),
            x=float(self._last.x + dx),
            y=float(self._last.y + dy),
            z=float(self._last.z + dz),
            yaw=yaw_norm,
            pitch=pitch_clamped,
            action_mask=int(action_mask),
        )

        self._last = nxt
        return nxt
