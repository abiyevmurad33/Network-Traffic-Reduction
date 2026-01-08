"""
Deterministic scripted controller for v1.0 logging.

Goals:
- Fully deterministic (Tier A reproducible).
- Exercises movement + turning + occasional attack.
- Produces a multi-binary button vector aligned with canonical action labels order.
- Avoids contradictory inputs (e.g., forward+back simultaneously).

This controller is open-loop: it does not use game state yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.logger.action_space import ActionSpace, canonical_action_space

# Canonical bit indices (must match ACTION_LABELS order)
IDX_MOVE_FORWARD = 0
IDX_MOVE_BACKWARD = 1
IDX_MOVE_LEFT = 2
IDX_MOVE_RIGHT = 3
IDX_TURN_LEFT = 4
IDX_TURN_RIGHT = 5
IDX_ATTACK = 6


@dataclass(frozen=True)
class Phase:
    """
    One phase of the scripted policy.

    duration_tics: number of tics to hold this phase
    pressed: list of indices (into the canonical action vector) that are True
    """

    duration_tics: int
    pressed: Sequence[int]

    def __post_init__(self) -> None:
        if self.duration_tics <= 0:
            raise ValueError(f"duration_tics must be > 0, got {self.duration_tics}")

        # Validate indices are unique and non-negative
        if len(set(self.pressed)) != len(self.pressed):
            raise ValueError(f"pressed indices must be unique, got {self.pressed!r}")
        if any(i < 0 for i in self.pressed):
            raise ValueError(f"pressed indices must be non-negative, got {self.pressed!r}")

        # Contradiction checks (keep strict for v1)
        pressed_set = set(self.pressed)
        if IDX_MOVE_FORWARD in pressed_set and IDX_MOVE_BACKWARD in pressed_set:
            raise ValueError("Phase cannot press MOVE_FORWARD and MOVE_BACKWARD together")
        if IDX_MOVE_LEFT in pressed_set and IDX_MOVE_RIGHT in pressed_set:
            raise ValueError("Phase cannot press MOVE_LEFT and MOVE_RIGHT together")
        if IDX_TURN_LEFT in pressed_set and IDX_TURN_RIGHT in pressed_set:
            raise ValueError("Phase cannot press TURN_LEFT and TURN_RIGHT together")


class ScriptedController:
    """
    Deterministic controller producing a 7D button vector per tic.

    Pattern design (cycle repeats):
    - Phase A: move forward + turn left
    - Phase B: strafe right + turn right
    - Phase C: move backward
    - Phase D: strafe left + turn left
    - Phase E: forward + short attack burst (attack toggles inside the phase)

    Attack bursts are deterministic (fixed on/off cadence).
    """

    def __init__(self, action_space: ActionSpace | None = None) -> None:
        self._space = action_space or canonical_action_space()

        if self._space.size != 7:
            raise ValueError(
                f"ScriptedController expects action space size 7, got {self._space.size}"
            )

        # Phase durations (tic-based, stable across machines)
        # 35 tics ≈ 1 second at ticrate=35
        self._phases: list[Phase] = [
            Phase(duration_tics=35, pressed=[IDX_MOVE_FORWARD, IDX_TURN_LEFT]),
            Phase(duration_tics=35, pressed=[IDX_MOVE_RIGHT, IDX_TURN_RIGHT]),
            Phase(duration_tics=28, pressed=[IDX_MOVE_BACKWARD]),
            Phase(duration_tics=28, pressed=[IDX_MOVE_LEFT, IDX_TURN_LEFT]),
            # Final phase includes deterministic attack burst logic.
            Phase(duration_tics=35, pressed=[IDX_MOVE_FORWARD]),
        ]
        self._cycle_len = sum(p.duration_tics for p in self._phases)

        # Attack burst cadence inside the last phase:
        # attack ON for 5 tics, OFF for 10 tics, repeat.
        self._attack_on = 5
        self._attack_off = 10
        self._attack_period = self._attack_on + self._attack_off

    @property
    def action_space(self) -> ActionSpace:
        return self._space

    @property
    def cycle_length_tics(self) -> int:
        return self._cycle_len

    def buttons_for_tic(self, tic: int) -> list[bool]:
        """
        Return the canonical 7-length boolean button vector for the given tic.

        tic: integer episode time (0..)
        """
        if tic < 0:
            raise ValueError(f"tic must be >= 0, got {tic}")

        # Find phase within cycle
        t = tic % self._cycle_len
        phase_idx, phase_offset = self._locate_phase(t)

        phase = self._phases[phase_idx]
        buttons = [False] * self._space.size
        for i in phase.pressed:
            buttons[i] = True

        # Add deterministic attack burst during the last phase only
        if phase_idx == len(self._phases) - 1:
            if (phase_offset % self._attack_period) < self._attack_on:
                buttons[IDX_ATTACK] = True

        # Final sanity: no contradictions
        self._assert_no_contradictions(buttons)

        return buttons

    def action_mask_for_tic(self, tic: int) -> int:
        """
        Convenience: return action_mask bitmask for the given tic.
        """
        return self._space.encode(self.buttons_for_tic(tic))

    def _locate_phase(self, t_in_cycle: int) -> tuple[int, int]:
        """
        Returns (phase_index, offset_within_phase).
        """
        acc = 0
        for i, phase in enumerate(self._phases):
            if acc <= t_in_cycle < acc + phase.duration_tics:
                return i, t_in_cycle - acc
            acc += phase.duration_tics

        # Should never happen
        raise RuntimeError("Failed to locate phase within cycle")

    @staticmethod
    def _assert_no_contradictions(buttons: Sequence[bool]) -> None:
        if buttons[IDX_MOVE_FORWARD] and buttons[IDX_MOVE_BACKWARD]:
            raise AssertionError("Contradiction: forward+back")
        if buttons[IDX_MOVE_LEFT] and buttons[IDX_MOVE_RIGHT]:
            raise AssertionError("Contradiction: left+right")
        if buttons[IDX_TURN_LEFT] and buttons[IDX_TURN_RIGHT]:
            raise AssertionError("Contradiction: turn_left+turn_right")


def _self_test() -> None:
    ctl = ScriptedController()

    # Basic shape invariants
    for tic in [0, 1, 10, 34, 35, 70, 120, 999]:
        buttons = ctl.buttons_for_tic(tic)
        assert isinstance(buttons, list)
        assert len(buttons) == 7
        # Mask must validate via encode/decode round-trip
        mask = ctl.action_mask_for_tic(tic)
        decoded = ctl.action_space.decode(mask)
        assert decoded == buttons

    # Attack burst exists in last phase
    # We probe a tic known to be inside the last phase:
    # last phase starts at sum of first 4 durations
    last_phase_start = 35 + 35 + 28 + 28
    b0 = ctl.buttons_for_tic(last_phase_start + 0)
    b6 = ctl.buttons_for_tic(last_phase_start + 6)
    assert b0[IDX_ATTACK] is True  # within ON window
    assert b6[IDX_ATTACK] is False  # likely in OFF window

    print("controller.py self-test: OK")


if __name__ == "__main__":
    _self_test()
