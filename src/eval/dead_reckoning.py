from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.utils.angles import circular_abs_error_deg, circular_diff_deg


@dataclass(frozen=True)
class State:
    tic: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


@dataclass(frozen=True)
class DRConfig:
    update_interval_tics: int
    pos_eps: float
    yaw_eps_deg: float
    pitch_eps_deg: float


@dataclass(frozen=True)
class DRSummary:
    n_tics: int
    baseline_packets: int
    dr_packets: int
    periodic_packets: int
    correction_packets: int
    savings_ratio: float

    pos_mae: float
    yaw_mae_deg: float
    pitch_mae_deg: float

    pos_max: float
    yaw_max_deg: float
    pitch_max_deg: float


def _euclid3(dx: float, dy: float, dz: float) -> float:
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _estimate_rates(
    prev_auth: State | None, last_auth: State
) -> tuple[float, float, float, float, float]:
    """
    Estimate constant rates using the two most recent authoritative states.
    Returns: vx, vy, vz, yaw_rate_deg_per_tic, pitch_rate_deg_per_tic
    If prev_auth is None or dt<=0, returns zeros.
    """
    if prev_auth is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    dt = last_auth.tic - prev_auth.tic
    if dt <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    vx = (last_auth.x - prev_auth.x) / dt
    vy = (last_auth.y - prev_auth.y) / dt
    vz = (last_auth.z - prev_auth.z) / dt

    dyaw = circular_diff_deg(last_auth.yaw, prev_auth.yaw)
    dpitch = circular_diff_deg(last_auth.pitch, prev_auth.pitch)

    yaw_rate = dyaw / dt
    pitch_rate = dpitch / dt

    return vx, vy, vz, yaw_rate, pitch_rate


def _predict(last_auth: State, rates: tuple[float, float, float, float, float], tic: int) -> State:
    vx, vy, vz, yaw_rate, pitch_rate = rates
    dt = tic - last_auth.tic
    if dt < 0:
        dt = 0

    x = last_auth.x + vx * dt
    y = last_auth.y + vy * dt
    z = last_auth.z + vz * dt
    yaw = last_auth.yaw + yaw_rate * dt
    pitch = last_auth.pitch + pitch_rate * dt

    return State(tic=tic, x=x, y=y, z=z, yaw=yaw, pitch=pitch)


def simulate_dead_reckoning(states: Sequence[State], cfg: DRConfig) -> DRSummary:
    """
    Simulate DR with periodic updates and threshold-based corrections.

    - Periodic update is attempted every cfg.update_interval_tics since last send.
    - Correction is sent immediately if errors exceed thresholds.
    """
    if cfg.update_interval_tics <= 0:
        raise ValueError("update_interval_tics must be > 0")
    if len(states) < 2:
        raise ValueError("Need at least 2 states to evaluate DR")

    # Ensure strict tic monotonicity
    for i in range(1, len(states)):
        if states[i].tic <= states[i - 1].tic:
            raise ValueError("States must be strictly increasing by tic")

    n = len(states)
    baseline_packets = n  # "no prediction": full state each tic

    # Authority tracking
    prev_auth: State | None = None
    last_auth: State = states[0]
    last_send_tic: int = states[0].tic

    periodic_packets = 1  # initial authoritative packet
    correction_packets = 0

    rates = _estimate_rates(prev_auth, last_auth)

    # Error accumulation over predicted-only tics
    pos_err_sum = 0.0
    yaw_err_sum = 0.0
    pitch_err_sum = 0.0
    err_count = 0

    pos_max = 0.0
    yaw_max = 0.0
    pitch_max = 0.0

    for i in range(1, n):
        gt = states[i]

        # Predict from last authoritative state
        pred = _predict(last_auth, rates, gt.tic)

        dx = pred.x - gt.x
        dy = pred.y - gt.y
        dz = pred.z - gt.z
        pos_err = _euclid3(dx, dy, dz)
        yaw_err = circular_abs_error_deg(pred.yaw, gt.yaw)
        pitch_err = circular_abs_error_deg(pred.pitch, gt.pitch)

        pos_err_sum += pos_err
        yaw_err_sum += yaw_err
        pitch_err_sum += pitch_err
        err_count += 1

        pos_max = max(pos_max, pos_err)
        yaw_max = max(yaw_max, yaw_err)
        pitch_max = max(pitch_max, pitch_err)

        need_periodic = (gt.tic - last_send_tic) >= cfg.update_interval_tics
        need_correction = (
            pos_err > cfg.pos_eps or yaw_err > cfg.yaw_eps_deg or pitch_err > cfg.pitch_eps_deg
        )

        if need_correction:
            # Send correction now (authoritative reset at this tic)
            correction_packets += 1
            last_send_tic = gt.tic
            prev_auth = last_auth
            last_auth = gt
            rates = _estimate_rates(prev_auth, last_auth)
            continue

        if need_periodic:
            periodic_packets += 1
            last_send_tic = gt.tic
            prev_auth = last_auth
            last_auth = gt
            rates = _estimate_rates(prev_auth, last_auth)

    dr_packets = periodic_packets + correction_packets
    savings_ratio = 1.0 - (dr_packets / baseline_packets)

    pos_mae = pos_err_sum / max(err_count, 1)
    yaw_mae = yaw_err_sum / max(err_count, 1)
    pitch_mae = pitch_err_sum / max(err_count, 1)

    return DRSummary(
        n_tics=n,
        baseline_packets=baseline_packets,
        dr_packets=dr_packets,
        periodic_packets=periodic_packets,
        correction_packets=correction_packets,
        savings_ratio=savings_ratio,
        pos_mae=pos_mae,
        yaw_mae_deg=yaw_mae,
        pitch_mae_deg=pitch_mae,
        pos_max=pos_max,
        yaw_max_deg=yaw_max,
        pitch_max_deg=pitch_max,
    )
