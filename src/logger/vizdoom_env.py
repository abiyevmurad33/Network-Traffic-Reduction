"""
ViZDoom environment wrapper for v1.0 logging.

Key design choices (based on probe results):
- Pose is extracted via GameVariables:
  POSITION_X, POSITION_Y, POSITION_Z, ANGLE, PITCH
- We add these variables programmatically to avoid relying on scenario configs exposing them.
- We validate the available buttons order matches our canonical action space order.

This module does NOT write logs to disk; it provides a clean iterator over tics.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import vizdoom as vzd

from src.logger.action_space import ACTION_LABELS, ActionSpace, canonical_action_space
from src.logger.controller import ScriptedController
from src.utils.angles import normalize_signed_deg


@dataclass(frozen=True)
class Pose:
    pos_x: float
    pos_y: float
    pos_z: float
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class TicSample:
    session_id: str
    tic: int
    player_id: str
    pose: Pose
    action_mask: int
    # Action vector as sent to ViZDoom (aligned with game.get_available_buttons())
    action_vector: list[int]


class VizDoomEnv:
    """
    Wrapper around vzd.DoomGame that:
    - loads a scenario cfg,
    - ensures required GameVariables exist,
    - runs a per-tic loop advancing exactly 1 tic per action.

    Intended usage:
      env = VizDoomEnv(cfg_path=..., iwad_required_path=...)
      env.init(window_visible=True, seed=12345)
      for sample in env.run(session_id=..., controller=..., max_tics=...):
          ...
      env.close()
    """

    REQUIRED_VAR_NAMES = ["POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "PITCH"]

    def __init__(self, cfg_path: str, iwad_required_path: str = "freedoom2.wad") -> None:
        self.cfg_path = cfg_path
        self.iwad_required_path = iwad_required_path

        self._game: vzd.DoomGame | None = None
        self._action_space: ActionSpace = canonical_action_space()

        # Indices into state.game_variables in the order we add them.
        self._gv_idx: dict[str, int] = {}

    @property
    def game(self) -> vzd.DoomGame:
        if self._game is None:
            raise RuntimeError("VizDoomEnv not initialized. Call init() first.")
        return self._game

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def init(self, window_visible: bool = True, seed: int | None = None) -> None:
        """
        Initialize the DoomGame instance.

        - Validates required files exist.
        - Adds required GameVariables for pose.
        - Validates available buttons order matches canonical labels.
        """
        self._validate_files()

        game = vzd.DoomGame()
        game.load_config(self.cfg_path)

        # Visibility override (must happen before init)
        if hasattr(game, "set_window_visible"):
            game.set_window_visible(window_visible)

        # Best-effort seeding (API availability can vary)
        if seed is not None and hasattr(game, "set_seed"):
            try:
                game.set_seed(int(seed))
            except Exception:
                # If set_seed exists but fails, we treat as non-fatal for now.
                # Determinism can still be achieved via scripted controller and fixed scenario.
                pass

        # Enable required pose variables programmatically
        self._enable_required_game_variables(game)

        # Initialize engine
        game.init()

        # Validate available buttons match our canonical action label order
        self._validate_buttons_order(game)

        # After init, snapshot indices for state.game_variables vector
        self._build_game_variable_index(game)

        self._game = game

    def close(self) -> None:
        if self._game is not None:
            try:
                self._game.close()
            finally:
                self._game = None

    def new_episode(self) -> None:
        self.game.new_episode()

    def get_tics_per_second(self) -> float | None:
        """
        Best-effort: return ticrate if exposed. If not, caller uses protocol default (35).
        """
        if hasattr(self.game, "get_ticrate"):
            try:
                return float(self.game.get_ticrate())
            except Exception:
                return None
        return None

    def run(
        self,
        session_id: str,
        controller: ScriptedController,
        max_tics: int,
        player_id: str = "p1",
    ) -> Iterator[TicSample]:
        """
        Run a fresh episode and yield TicSample for each tic.

        Logging convention:
        - Read state at tic t
        - Compute action for tic t (based on tic)
        - Yield sample containing state-at-t and action-at-t
        - Apply action to advance exactly 1 tic
        """
        if max_tics <= 0:
            raise ValueError(f"max_tics must be > 0, got {max_tics}")

        self.new_episode()

        # Determine action vector size from game (should match canonical size=7)
        btns = self.game.get_available_buttons()
        action_len = len(btns)
        if action_len != self._action_space.size:
            raise RuntimeError(
                f"Available buttons size {action_len} does not match action space size {self._action_space.size}"
            )

        for _ in range(max_tics):
            tic = int(self.game.get_episode_time())

            st = self.game.get_state()
            if st is None:
                raise RuntimeError("game.get_state() returned None during episode")

            pose = self._read_pose_from_state(st)

            buttons_bool = controller.buttons_for_tic(tic)
            action_mask = self._action_space.encode(buttons_bool)

            # Convert to ViZDoom action vector order (validated to match canonical labels)
            action_vec = [1 if b else 0 for b in buttons_bool]

            yield TicSample(
                session_id=session_id,
                tic=tic,
                player_id=player_id,
                pose=pose,
                action_mask=action_mask,
                action_vector=action_vec,
            )

            # Advance exactly 1 tic
            self.game.make_action(action_vec, 1)

            if self.game.is_episode_finished():
                break

    # -------------------------
    # Internal helpers
    # -------------------------

    def _validate_files(self) -> None:
        # cfg must exist
        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"Scenario config not found: {self.cfg_path}")

        # IWAD required file must exist (we enforce presence in repo root)
        if not os.path.exists(self.iwad_required_path):
            raise FileNotFoundError(
                f"Required IWAD not found at {self.iwad_required_path!r}. "
                "Expected a local file (e.g., repo root freedoom2.wad)."
            )

    def _enable_required_game_variables(self, game: vzd.DoomGame) -> None:
        for name in self.REQUIRED_VAR_NAMES:
            if not hasattr(vzd.GameVariable, name):
                raise RuntimeError(f"ViZDoom build missing required GameVariable: {name}")
            gv = getattr(vzd.GameVariable, name)
            game.add_available_game_variable(gv)

    def _build_game_variable_index(self, game: vzd.DoomGame) -> None:
        # In ViZDoom, state.game_variables follows the order of available game variables.
        available = game.get_available_game_variables()
        names = [str(v).split(".")[-1] for v in available]  # e.g., "POSITION_X"
        idx: dict[str, int] = {}
        for i, n in enumerate(names):
            idx[n] = i

        for req in self.REQUIRED_VAR_NAMES:
            if req not in idx:
                raise RuntimeError(
                    f"Required game variable {req} not present in available list: {names}"
                )

        self._gv_idx = idx

    def _validate_buttons_order(self, game: vzd.DoomGame) -> None:
        btns = game.get_available_buttons()
        # Convert Button enum to name strings like "MOVE_FORWARD"
        btn_names = [str(b).split(".")[-1] for b in btns]

        if btn_names != ACTION_LABELS:
            raise RuntimeError(
                "Available buttons order does not match canonical action space.\n"
                f"Expected: {ACTION_LABELS}\n"
                f"Got:      {btn_names}\n"
                "Fix by updating scenarios/basic_movement.cfg available_buttons order "
                "or adjusting ACTION_LABELS (not recommended)."
            )

    def _read_pose_from_state(self, st: vzd.GameState) -> Pose:
        gv = getattr(st, "game_variables", None)
        if gv is None:
            raise RuntimeError("State has no game_variables; pose extraction unavailable")

        # Extract in our required names via index map
        px = float(gv[self._gv_idx["POSITION_X"]])
        py = float(gv[self._gv_idx["POSITION_Y"]])
        pz = float(gv[self._gv_idx["POSITION_Z"]])
        yaw = float(gv[self._gv_idx["ANGLE"]])
        pitch = float(gv[self._gv_idx["PITCH"]])

        # Normalize to canonical range (-180, 180]
        yaw_n = normalize_signed_deg(yaw)
        pitch_n = normalize_signed_deg(pitch)

        return Pose(pos_x=px, pos_y=py, pos_z=pz, yaw_deg=yaw_n, pitch_deg=pitch_n)


def _self_test() -> None:
    """
    Minimal integration test (requires freedoom2.wad + scenarios cfg/wad present).
    Prints a few samples and exits.
    """
    env = VizDoomEnv(cfg_path=os.path.join("scenarios", "basic_movement.cfg"))
    try:
        env.init(window_visible=True, seed=12345)
        controller = ScriptedController(space := canonical_action_space())
        # Run 5 tics only
        it = env.run(session_id="selftest", controller=controller, max_tics=5, player_id="p1")
        for s in it:
            print(
                f"tic={s.tic} pos=({s.pose.pos_x:.1f},{s.pose.pos_y:.1f},{s.pose.pos_z:.1f}) "
                f"yaw={s.pose.yaw_deg:.1f} pitch={s.pose.pitch_deg:.1f} mask={s.action_mask}"
            )
        print("vizdoom_env.py self-test: OK")
    finally:
        env.close()


if __name__ == "__main__":
    _self_test()
