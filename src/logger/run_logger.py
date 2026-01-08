from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.logger.action_space import canonical_action_space
from src.logger.controller import ScriptedController
from src.logger.vizdoom_env import VizDoomEnv
from src.utils.paths import atomic_replace, build_session_paths, ensure_dir, repo_root_from_file
from src.utils.time_id import make_session_id, utc_now_iso

CSV_HEADER = [
    "session_id",
    "tic",
    "player_id",
    "pos_x",
    "pos_y",
    "pos_z",
    "yaw_deg",
    "pitch_deg",
    "action_mask",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a deterministic ViZDoom logging session (v1).")

    p.add_argument(
        "--scenario-config",
        type=str,
        default=str(Path("scenarios") / "basic_movement.cfg"),
        help="Path to ViZDoom scenario cfg (default: scenarios/basic_movement.cfg).",
    )
    p.add_argument(
        "--duration-seconds",
        type=float,
        default=120.0,
        help="Target duration in seconds (default: 120).",
    )
    p.add_argument(
        "--protocol-tier",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Protocol tier: A=reproducible (seeded), B=exploration (default: A).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for Tier A (default: 12345). For Tier B, seed is ignored unless provided.",
    )
    p.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional session id; if omitted, generated per protocol.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(Path("data") / "raw_logs"),
        help="Output root directory (default: data/raw_logs).",
    )

    vis = p.add_mutually_exclusive_group()
    vis.add_argument(
        "--window-visible",
        dest="window_visible",
        action="store_true",
        help="Show ViZDoom window (default).",
    )
    vis.add_argument(
        "--headless",
        dest="window_visible",
        action="store_false",
        help="Run headless (window hidden).",
    )
    p.set_defaults(window_visible=True)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = repo_root_from_file(__file__)
    scenario_cfg = (repo_root / Path(args.scenario_config)).resolve()
    output_root = (repo_root / Path(args.output_root)).resolve()
    iwad_path = (repo_root / "freedoom2.wad").resolve()

    session_id = args.session_id.strip() or make_session_id()

    # Tier rules
    if args.protocol_tier == "A":
        seed = int(args.seed)
    else:
        seed = None if args.seed is None else int(args.seed)

    # Compute target tics
    # Prefer env-reported ticrate if available, otherwise protocol default (35).
    tics_per_second_default = 35.0
    target_tics = int(round(args.duration_seconds * tics_per_second_default))

    sp = build_session_paths(output_root, session_id)
    ensure_dir(sp.session_dir)

    # Prepare controller and env
    action_space = canonical_action_space()
    controller = ScriptedController(action_space)

    env = VizDoomEnv(cfg_path=str(scenario_cfg), iwad_required_path=str(iwad_path))

    # We will only rename tmp->final on success
    wrote_any_rows = False
    first_tic: int | None = None
    last_tic: int | None = None
    ended_early = False
    end_reason = ""
    tics_recorded = 0
    tics_per_second_used = tics_per_second_default

    try:
        env.init(window_visible=bool(args.window_visible), seed=seed)
        tps = env.get_tics_per_second()
        if tps is not None and tps > 0:
            tics_per_second_used = float(tps)
            target_tics = int(round(args.duration_seconds * tics_per_second_used))

        # Stream-write CSV to tmp
        with sp.states_csv_tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)

            for sample in env.run(
                session_id=session_id,
                controller=controller,
                max_tics=target_tics,
                player_id="p1",
            ):
                if first_tic is None:
                    first_tic = sample.tic
                last_tic = sample.tic

                w.writerow(
                    [
                        sample.session_id,
                        sample.tic,
                        sample.player_id,
                        f"{sample.pose.pos_x:.6f}",
                        f"{sample.pose.pos_y:.6f}",
                        f"{sample.pose.pos_z:.6f}",
                        f"{sample.pose.yaw_deg:.6f}",
                        f"{sample.pose.pitch_deg:.6f}",
                        sample.action_mask,
                    ]
                )
                wrote_any_rows = True
                tics_recorded += 1

        # Determine early termination
        if tics_recorded < target_tics:
            ended_early = True
            end_reason = "episode_finished_before_target_tics"

        # Write session.json to tmp
        scenario_name = Path(args.scenario_config).stem  # e.g., basic_movement
        session_meta = {
            "schema_version": "1.0",
            "session_id": session_id,
            "created_utc": utc_now_iso(),
            "engine": "vizdoom",
            "scenario_name": scenario_name,
            "scenario_config_path": str(Path(args.scenario_config)).replace("\\", "/"),
            "wad": "freedoom2.wad",
            "seed": seed,
            "tics_per_second": tics_per_second_used,
            "duration_seconds": float(args.duration_seconds),
            "protocol_tier": args.protocol_tier,
            "warmup_seconds": 0,
            "players": [
                {"player_id": "p1", "controller": "script", "notes": ""},
            ],
            "action_space": action_space.to_metadata(),
            "ended_early": ended_early,
            "tics_recorded": tics_recorded,
            "end_reason": end_reason,
            "first_tic": first_tic,
            "last_tic": last_tic,
        }

        with sp.session_json_tmp.open("w", encoding="utf-8") as jf:
            json.dump(session_meta, jf, indent=2)

        # Basic safety: do not finalize empty outputs
        if not wrote_any_rows:
            raise RuntimeError("No rows were recorded; refusing to finalize outputs.")

        # Atomic finalize
        atomic_replace(sp.states_csv_tmp, sp.states_csv)
        atomic_replace(sp.session_json_tmp, sp.session_json)

        print("OK")
        print("session_id:", session_id)
        print("output_dir:", str(sp.session_dir))
        print("tics_recorded:", tics_recorded)
        return 0

    except Exception as e:
        # Leave .tmp files for debugging; do not produce ambiguous final outputs.
        print("FAILED:", type(e).__name__, str(e), file=sys.stderr)
        print("tmp_outputs:", str(sp.states_csv_tmp), str(sp.session_json_tmp), file=sys.stderr)
        return 2

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
