from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import vizdoom as vzd


def _safe_call(obj: Any, name: str, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
    """Call obj.name(*args, **kwargs) if it exists; return (ok, result_or_err_str)."""
    if not hasattr(obj, name):
        return False, f"missing: {name}"
    try:
        return True, getattr(obj, name)(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "scenarios", "basic_movement.cfg")
    iwad_path = os.path.join(repo_root, "freedoom2.wad")

    _print_header("ENVIRONMENT")
    print("UTC now:", datetime.now(UTC).isoformat())
    print("Python:", sys.version)
    print("Repo root:", repo_root)
    print("Config path exists:", os.path.exists(cfg_path), cfg_path)
    print("IWAD exists:", os.path.exists(iwad_path), iwad_path)

    _print_header("VIZDOOM VERSION")
    print("vizdoom module:", vzd.__file__)
    print("vizdoom version:", getattr(vzd, "__version__", "version-unknown"))

    _print_header("INITIALIZE GAME")
    game = vzd.DoomGame()
    game.load_config(cfg_path)

    # Enable the minimal pose variables required by our schema.
    # We do this programmatically because the scenario currently exposes none.
    required_var_names = ["POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "PITCH"]

    _print_header("ENABLE REQUIRED GAME VARIABLES (BEST-EFFORT)")
    enabled = []
    missing = []
    for name in required_var_names:
        if hasattr(vzd.GameVariable, name):
            gv = getattr(vzd.GameVariable, name)
            try:
                game.add_available_game_variable(gv)
                enabled.append(name)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to add {name}: {type(e).__name__}: {e}")
        else:
            missing.append(name)

    print("Enabled:", enabled)
    print("Missing:", missing)

    # Keep config defaults (e.g., window_visible) to avoid mismatches.
    try:
        game.init()
    except Exception as e:  # noqa: BLE001
        print("FAILED: game.init()")
        print(type(e).__name__ + ":", e)
        return 2

    print("OK: game.init()")

    _print_header("AVAILABLE BUTTONS")
    ok_btns, buttons = _safe_call(game, "get_available_buttons")
    print("get_available_buttons:", ok_btns, buttons)
    ok_size, btn_size = _safe_call(game, "get_available_buttons_size")
    print("get_available_buttons_size:", ok_size, btn_size)

    _print_header("AVAILABLE GAME VARIABLES (IF EXPOSED)")
    ok_vars, vars_list = _safe_call(game, "get_available_game_variables")
    print("get_available_game_variables:", ok_vars, vars_list)

    # GameVariable enum iteration is not supported in some pybind builds (e.g., ViZDoom 1.2.4 on Windows).
    # We therefore only probe variables if ViZDoom explicitly reports available game variables.
    _print_header("PROBE: READ AVAILABLE GAME VARIABLES (BEST-EFFORT)")
    if ok_vars and isinstance(vars_list, list) and len(vars_list) > 0:
        readable: list[tuple[str, Any]] = []
        for gv in vars_list:
            try:
                val = game.get_game_variable(gv)
                readable.append((str(gv), val))
            except Exception:
                continue

        print("Readable available game variables count:", len(readable))
        for name, val in readable:
            print(f"{name} = {val}")
    else:
        print(
            "No available game variables reported by this scenario; skipping GameVariable probing."
        )

    _print_header("RUN SHORT EPISODE AND INSPECT STATE")
    game.new_episode()

    # Determine action vector length
    action_len = 1
    if isinstance(btn_size, int) and btn_size > 0:
        action_len = btn_size

    noop = [0] * action_len

    snapshots: list[dict[str, Any]] = []
    for step in range(3):
        # Inspect state BEFORE advancing
        st = game.get_state()
        if st is None:
            print(f"step={step}: state=None")
        else:
            print(f"step={step}: state type={type(st).__name__}")

            # game_variables array (if present)
            if hasattr(st, "game_variables") and st.game_variables is not None:
                try:
                    gv_shape = getattr(st.game_variables, "shape", None)
                    print("  state.game_variables shape:", gv_shape)
                except Exception as e:  # noqa: BLE001
                    print("  state.game_variables read error:", type(e).__name__, e)

            # objects (if present)
            if hasattr(st, "objects") and st.objects is not None:
                try:
                    print("  state.objects count:", len(st.objects))
                    # Print first few objects with key fields
                    for obj in st.objects[:10]:
                        d = {
                            "name": getattr(obj, "name", None),
                            "id": getattr(obj, "id", None),
                            "position_x": getattr(obj, "position_x", None),
                            "position_y": getattr(obj, "position_y", None),
                            "position_z": getattr(obj, "position_z", None),
                            "angle": getattr(obj, "angle", None),
                            "pitch": getattr(obj, "pitch", None),
                            "velocity_x": getattr(obj, "velocity_x", None),
                            "velocity_y": getattr(obj, "velocity_y", None),
                            "velocity_z": getattr(obj, "velocity_z", None),
                        }
                        print("   ", d)
                except Exception as e:  # noqa: BLE001
                    print("  state.objects read error:", type(e).__name__, e)

            # Save a minimal snapshot for reference
            snapshots.append(
                {
                    "step": step,
                    "episode_time": game.get_episode_time(),
                    "has_game_variables": bool(getattr(st, "game_variables", None) is not None),
                    "has_objects": bool(getattr(st, "objects", None) is not None),
                }
            )

        # Advance exactly 1 tic
        game.make_action(noop, 1)

    _print_header("SUMMARY SNAPSHOTS (JSON)")
    print(json.dumps(snapshots, indent=2))

    game.close()
    print("\nOK: probe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
