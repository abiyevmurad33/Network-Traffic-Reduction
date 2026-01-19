from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionEncoding = Literal["mask_bits8"]

TARGET_MODE = Literal["delta"]


@dataclass(frozen=True)
class FeatureSpec:
    version: int
    action_encoding: ActionEncoding
    include_pose: bool = True  # x,y,z,yaw,pitch

    @property
    def dim(self) -> int:
        # pose(5) + bits8(8) = 13
        return 13


@dataclass(frozen=True)
class TargetSpec:
    version: int
    mode: TARGET_MODE = "delta"

    @property
    def dim(self) -> int:
        # dx,dy,dz,dyaw,dpitch
        return 5


FEATURE_SPEC_V1 = FeatureSpec(version=1, action_encoding="mask_bits8")
TARGET_SPEC_V1 = TargetSpec(version=1, mode="delta")


def mask_to_bits8(mask: int) -> np.ndarray:
    # [ATTACK, USE, JUMP, SHIFT, W, S, A, D]
    return np.array(
        [
            1 if (mask & 1) else 0,
            1 if (mask & 2) else 0,
            1 if (mask & 4) else 0,
            1 if (mask & 256) else 0,
            1 if (mask & 8192) else 0,
            1 if (mask & 4096) else 0,
            1 if (mask & 2048) else 0,
            1 if (mask & 1024) else 0,
        ],
        dtype=np.float32,
    )
