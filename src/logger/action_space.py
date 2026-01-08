"""
Action space definition and bitmask encoding/decoding.

Canonical v1 action order (bit positions) — MUST remain stable for comparability:
Bit 0: MOVE_FORWARD
Bit 1: MOVE_BACKWARD
Bit 2: MOVE_LEFT
Bit 3: MOVE_RIGHT
Bit 4: TURN_LEFT
Bit 5: TURN_RIGHT
Bit 6: ATTACK

The logger stores actions as:
- action_mask: int bitmask with bit i representing whether ACTION_LABELS[i] is active.

This module is intentionally independent of ViZDoom APIs; it only defines encoding rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ACTION_LABELS: list[str] = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
]


@dataclass(frozen=True)
class ActionSpace:
    """
    Represents a multi-binary action space encoded as an integer bitmask.

    labels: ordered list where index == bit position.
    """

    labels: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.labels) == 0:
            raise ValueError("ActionSpace.labels must be non-empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"ActionSpace.labels must be unique, got {self.labels!r}")

    @property
    def size(self) -> int:
        return len(self.labels)

    def encode(self, buttons: Sequence[bool]) -> int:
        """
        Encode a boolean button vector to an integer bitmask.
        buttons[i] corresponds to labels[i].

        Raises:
          - ValueError if length mismatch
        """
        if len(buttons) != self.size:
            raise ValueError(
                f"buttons length {len(buttons)} does not match action space size {self.size}"
            )

        mask = 0
        for i, pressed in enumerate(buttons):
            if pressed:
                mask |= 1 << i
        return mask

    def decode(self, mask: int) -> list[bool]:
        """
        Decode an integer bitmask to a boolean button vector of length `size`.

        Raises:
          - ValueError if mask is negative
        """
        if mask < 0:
            raise ValueError(f"mask must be non-negative, got {mask}")
        return [bool(mask & (1 << i)) for i in range(self.size)]

    def validate_mask(self, mask: int) -> None:
        """
        Validate that mask is non-negative and does not use bits beyond the space size.

        Raises:
          - ValueError on invalid mask
        """
        if mask < 0:
            raise ValueError(f"mask must be non-negative, got {mask}")
        max_valid = (1 << self.size) - 1
        if mask > max_valid:
            raise ValueError(
                f"mask {mask} uses bits outside action space (max {max_valid} for size={self.size})"
            )

    def to_metadata(self) -> dict:
        """
        Serialize a minimal action_space object for session.json.
        """
        return {"type": "multi_binary_bitmask", "labels": list(self.labels)}


def canonical_action_space() -> ActionSpace:
    """
    Return the canonical v1 action space (fixed label order).
    """
    return ActionSpace(labels=ACTION_LABELS)


def mask_from_pressed_labels(pressed: Iterable[str], space: ActionSpace | None = None) -> int:
    """
    Convenience: build a mask from a set/list of pressed label strings.

    Example:
      mask_from_pressed_labels(["MOVE_FORWARD", "ATTACK"]) -> 1 + 64 = 65

    Raises:
      - ValueError if unknown label encountered
    """
    sp = space or canonical_action_space()
    label_to_bit = {lab: i for i, lab in enumerate(sp.labels)}

    mask = 0
    for lab in pressed:
        if lab not in label_to_bit:
            raise ValueError(f"unknown action label: {lab!r}")
        mask |= 1 << label_to_bit[lab]
    sp.validate_mask(mask)
    return mask


def _self_test() -> None:
    sp = canonical_action_space()
    assert sp.size == 7

    # Encode/decode round-trip
    buttons = [True, False, False, True, False, True, False]
    m = sp.encode(buttons)
    sp.validate_mask(m)
    assert sp.decode(m) == buttons

    # Known mask check: MOVE_FORWARD (bit0) + ATTACK (bit6) = 1 + 64 = 65
    m2 = mask_from_pressed_labels(["MOVE_FORWARD", "ATTACK"], sp)
    assert m2 == 65
    assert sp.decode(m2)[0] is True
    assert sp.decode(m2)[6] is True

    # Out-of-range bit must fail for size=7 (bit 7 would be 128)
    try:
        sp.validate_mask(128)
        raise AssertionError("Expected validate_mask to fail for out-of-range bits")
    except ValueError:
        pass

    print("action_space.py self-test: OK")


if __name__ == "__main__":
    _self_test()
