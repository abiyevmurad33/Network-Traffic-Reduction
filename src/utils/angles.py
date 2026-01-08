"""
Angle utilities for the thesis project.

Angle convention (v1):
- All angles are in degrees.
- Canonical normalization range is (-180, 180].

Why (-180, 180]?
- It reduces discontinuities for error metrics and learning targets.
- It supports stable circular difference calculations.

This module intentionally has no external dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

_EPS = 1e-12


def normalize_deg(angle_deg: float) -> float:
    """
    Normalize a degree angle to the range [0, 360).

    This is sometimes useful for display/serialization, but the project's canonical
    range for yaw/pitch is (-180, 180]. Use normalize_signed_deg() for that.
    """
    if not math.isfinite(angle_deg):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
    x = angle_deg % 360.0
    # Ensure 360.0 wraps to 0.0
    if abs(x - 360.0) < _EPS:
        x = 0.0
    return x


def normalize_signed_deg(angle_deg: float) -> float:
    """
    Normalize a degree angle to the canonical range (-180, 180].

    Notes:
    - 180 is included, -180 is excluded.
    - If the input is 540, output is 180.
    - If the input is -540, output is 180.
    """
    if not math.isfinite(angle_deg):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")

    # First normalize to [0, 360)
    x = normalize_deg(angle_deg)

    # Map to (-180, 180]
    if x > 180.0:
        x -= 360.0

    # Enforce the half-open endpoint rule: (-180, 180]
    # If x is -180 (possible if angle_deg was exactly -180 mod 360),
    # convert it to +180.
    if abs(x + 180.0) < _EPS:
        x = 180.0

    return x


def circular_diff_deg(a_deg: float, b_deg: float) -> float:
    """
    Smallest signed difference a - b in degrees, returned in (-180, 180].

    Example:
    - a=179, b=-179 => diff = -2 (not 358)
    """
    a_n = normalize_signed_deg(a_deg)
    b_n = normalize_signed_deg(b_deg)
    d = a_n - b_n
    return normalize_signed_deg(d)


def circular_abs_error_deg(a_deg: float, b_deg: float) -> float:
    """
    Absolute circular error between two angles, in [0, 180].
    """
    return abs(circular_diff_deg(a_deg, b_deg))


def circular_mean_deg(angles_deg: Iterable[float]) -> float:
    """
    Circular mean of angles in degrees, returned in (-180, 180].

    Raises:
    - ValueError if iterable is empty
    - ValueError if any angle is non-finite
    """
    sin_sum = 0.0
    cos_sum = 0.0
    n = 0

    for a in angles_deg:
        if not math.isfinite(a):
            raise ValueError(f"all angles must be finite, got {a!r}")
        rad = math.radians(a)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
        n += 1

    if n == 0:
        raise ValueError("angles_deg must be non-empty")

    mean_rad = math.atan2(sin_sum / n, cos_sum / n)
    mean_deg = math.degrees(mean_rad)
    return normalize_signed_deg(mean_deg)


def _self_test() -> None:
    # Normalization endpoint behavior
    assert normalize_signed_deg(180.0) == 180.0
    assert normalize_signed_deg(-180.0) == 180.0
    assert normalize_signed_deg(540.0) == 180.0
    assert normalize_signed_deg(-540.0) == 180.0

    # Basic wrapping
    assert normalize_signed_deg(181.0) == -179.0
    assert normalize_signed_deg(-181.0) == 179.0
    assert normalize_signed_deg(0.0) == 0.0
    assert normalize_signed_deg(360.0) == 0.0

    # Circular diff
    assert circular_diff_deg(179.0, -179.0) == -2.0
    assert circular_diff_deg(-179.0, 179.0) == 2.0

    # Abs error range
    assert circular_abs_error_deg(10.0, 10.0) == 0.0
    assert circular_abs_error_deg(10.0, 190.0) == 180.0

    # Circular mean sanity
    m = circular_mean_deg([179.0, -179.0])
    # mean should be around 180 (canonicalized), not around 0
    assert m == 180.0

    print("angles.py self-test: OK")


if __name__ == "__main__":
    _self_test()
