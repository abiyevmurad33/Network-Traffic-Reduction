from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.eval.dead_reckoning import DRConfig, State, simulate_dead_reckoning


def load_states(session_dir: Path) -> list[State]:
    states_path = session_dir / "states.csv"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing {states_path}")

    out: list[State] = []
    with states_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                State(
                    tic=int(row["tic"]),
                    x=float(row["pos_x"]),
                    y=float(row["pos_y"]),
                    z=float(row["pos_z"]),
                    yaw=float(row["yaw_deg"]),
                    pitch=float(row["pitch_deg"]),
                )
            )
    if len(out) < 2:
        raise ValueError("Need at least 2 rows in states.csv")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate Dead Reckoning baseline on a session.")
    p.add_argument("session_dir", type=str, help="Path to data/raw_logs/<session_id>/")
    p.add_argument("--interval", type=int, default=10, help="Periodic update interval in tics.")
    p.add_argument("--pos-eps", type=float, default=5.0, help="Position error threshold (units).")
    p.add_argument("--yaw-eps", type=float, default=5.0, help="Yaw error threshold (deg).")
    p.add_argument("--pitch-eps", type=float, default=5.0, help="Pitch error threshold (deg).")
    p.add_argument(
        "--out",
        type=str,
        default="evaluation_dr.json",
        help="Output filename (written inside session_dir).",
    )
    args = p.parse_args(argv)

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        print(f"FAIL: session_dir does not exist: {session_dir}", file=sys.stderr)
        return 2

    states = load_states(session_dir)
    cfg = DRConfig(
        update_interval_tics=int(args.interval),
        pos_eps=float(args.pos_eps),
        yaw_eps_deg=float(args.yaw_eps),
        pitch_eps_deg=float(args.pitch_eps),
    )

    summary = simulate_dead_reckoning(states, cfg)

    payload = {
        "method": "dead_reckoning_v1",
        "config": {
            "update_interval_tics": cfg.update_interval_tics,
            "pos_eps": cfg.pos_eps,
            "yaw_eps_deg": cfg.yaw_eps_deg,
            "pitch_eps_deg": cfg.pitch_eps_deg,
        },
        "summary": {
            "n_tics": summary.n_tics,
            "baseline_packets": summary.baseline_packets,
            "dr_packets": summary.dr_packets,
            "periodic_packets": summary.periodic_packets,
            "correction_packets": summary.correction_packets,
            "savings_ratio": summary.savings_ratio,
            "pos_mae": summary.pos_mae,
            "yaw_mae_deg": summary.yaw_mae_deg,
            "pitch_mae_deg": summary.pitch_mae_deg,
            "pos_max": summary.pos_max,
            "yaw_max_deg": summary.yaw_max_deg,
            "pitch_max_deg": summary.pitch_max_deg,
        },
    }

    out_path = session_dir / args.out
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("OK")
    print("session_dir:", str(session_dir))
    print("out:", str(out_path))
    print("savings_ratio:", summary.savings_ratio)
    print("dr_packets:", summary.dr_packets, "baseline_packets:", summary.baseline_packets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
