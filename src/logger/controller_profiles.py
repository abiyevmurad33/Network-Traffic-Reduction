from __future__ import annotations

from dataclasses import dataclass

from src.logger.action_space import ActionSpace


@dataclass(frozen=True)
class ControllerProfile:
    name: str


class BasicMoveProfile(ControllerProfile):
    def __init__(self) -> None:
        super().__init__(name="basic_move")


class StrafeTurnProfile(ControllerProfile):
    def __init__(self) -> None:
        super().__init__(name="strafe_turn")


class CombatBurstProfile(ControllerProfile):
    def __init__(self) -> None:
        super().__init__(name="combat_burst")


class ProfiledScriptedController:
    """
    Deterministic controller with named profiles.

    Output is a list[bool] aligned with ActionSpace.labels (canonical order).
    """

    def __init__(self, action_space: ActionSpace, profile: ControllerProfile) -> None:
        self.space = action_space
        self.profile = profile

        # Quick lookup indices
        self.idx = {label: i for i, label in enumerate(self.space.labels)}

    def buttons_for_tic(self, tic: int) -> list[bool]:
        btn = [False] * self.space.size

        if self.profile.name == "basic_move":
            # Phase loop: forward + turn left (baseline you already observed)
            # 0-149: move forward
            # 150-299: move forward + turn left
            # 300-449: move forward + turn right
            phase = tic % 450
            if phase < 150:
                btn[self.idx["MOVE_FORWARD"]] = True
            elif phase < 300:
                btn[self.idx["MOVE_FORWARD"]] = True
                btn[self.idx["TURN_LEFT"]] = True
            else:
                btn[self.idx["MOVE_FORWARD"]] = True
                btn[self.idx["TURN_RIGHT"]] = True
            return btn

        if self.profile.name == "strafe_turn":
            # More “FPS-like” micro-pattern:
            # 0-79: strafe left + turn right (circle strafe)
            # 80-159: strafe right + turn left (reverse circle)
            # 160-219: stop and do aim sweeps (turn left then right)
            # 220-279: forward burst + mild turn
            # repeat
            phase = tic % 280
            if phase < 80:
                btn[self.idx["MOVE_LEFT"]] = True
                btn[self.idx["TURN_RIGHT"]] = True
            elif phase < 160:
                btn[self.idx["MOVE_RIGHT"]] = True
                btn[self.idx["TURN_LEFT"]] = True
            elif phase < 190:
                btn[self.idx["TURN_LEFT"]] = True
            elif phase < 220:
                btn[self.idx["TURN_RIGHT"]] = True
            else:
                btn[self.idx["MOVE_FORWARD"]] = True
                # mild alternating yaw
                if (tic % 20) < 10:
                    btn[self.idx["TURN_LEFT"]] = True
                else:
                    btn[self.idx["TURN_RIGHT"]] = True
            return btn

        if self.profile.name == "combat_burst":
            # FPS-like proxy:
            # - circle strafe segments
            # - periodic micro-aim sweeps
            # - short ATTACK bursts (3 tics) every 10 tics during "engagement" windows
            #
            # Timeline (repeat every 350 tics):
            # 0-139: strafe left + turn right (engage) with bursts
            # 140-279: strafe right + turn left (engage) with bursts
            # 280-319: stop + aim sweep left/right (no attack)
            # 320-349: forward reposition + mild turn (no attack)
            phase = tic % 350

            def attack_burst() -> bool:
                # 3-tic burst every 10 tics
                return (tic % 10) in (0, 1, 2)

            if phase < 140:
                btn[self.idx["MOVE_LEFT"]] = True
                btn[self.idx["TURN_RIGHT"]] = True
                if attack_burst():
                    btn[self.idx["ATTACK"]] = True
            elif phase < 280:
                btn[self.idx["MOVE_RIGHT"]] = True
                btn[self.idx["TURN_LEFT"]] = True
                if attack_burst():
                    btn[self.idx["ATTACK"]] = True
            elif phase < 300:
                btn[self.idx["TURN_LEFT"]] = True
            elif phase < 320:
                btn[self.idx["TURN_RIGHT"]] = True
            else:
                btn[self.idx["MOVE_FORWARD"]] = True
                if (tic % 20) < 10:
                    btn[self.idx["TURN_LEFT"]] = True
                else:
                    btn[self.idx["TURN_RIGHT"]] = True

            return btn

        raise ValueError(f"Unknown controller profile: {self.profile.name!r}")
