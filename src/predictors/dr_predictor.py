from __future__ import annotations

from dataclasses import dataclass

from src.predictors.base import Obs, Predictor
from src.utils.angles import circular_diff_deg


def _estimate_rates(prev: Obs | None, cur: Obs) -> tuple[float, float, float, float, float]:
    if prev is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    dt = cur.tic - prev.tic
    if dt <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    vx = (cur.x - prev.x) / dt
    vy = (cur.y - prev.y) / dt
    vz = (cur.z - prev.z) / dt

    dyaw = circular_diff_deg(cur.yaw, prev.yaw)
    dpitch = circular_diff_deg(cur.pitch, prev.pitch)

    return vx, vy, vz, dyaw / dt, dpitch / dt


@dataclass
class DRPredictor(Predictor):
    """
    Dead Reckoning predictor: constant velocity + constant yaw/pitch rate.
    Uses the last two observed/predicted states to estimate rates.
    """

    _prev: Obs | None = None
    _cur: Obs | None = None
    _rates: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._prev = None
        self._cur = None
        self._rates = (0.0, 0.0, 0.0, 0.0, 0.0)

    def observe(self, obs: Obs) -> None:
        # When we inject an authoritative state, treat it as the new current.
        if self._cur is None:
            self._cur = obs
            self._prev = None
            self._rates = (0.0, 0.0, 0.0, 0.0, 0.0)
            return

        # Shift and re-estimate rates from last two known states
        self._prev = self._cur
        self._cur = obs
        self._rates = _estimate_rates(self._prev, self._cur)

    def step(self, action_mask: int) -> Obs:
        # DR ignores action_mask; it extrapolates from motion only.
        if self._cur is None:
            raise RuntimeError("DRPredictor.step() called before observe().")

        vx, vy, vz, yaw_rate, pitch_rate = self._rates
        nxt = Obs(
            tic=self._cur.tic + 1,
            x=self._cur.x + vx,
            y=self._cur.y + vy,
            z=self._cur.z + vz,
            yaw=self._cur.yaw + yaw_rate,
            pitch=self._cur.pitch + pitch_rate,
            action_mask=action_mask,
        )

        # Advance internal state
        self._prev = self._cur
        self._cur = nxt
        self._rates = _estimate_rates(self._prev, self._cur)
        return nxt
