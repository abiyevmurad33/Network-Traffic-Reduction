from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.eval.net_sim import DRConfig, NetConfig, State, evaluate_dr_under_network


def load_states(session_dir: Path) -> list[State]:
    p = session_dir / "states.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

    out: list[State] = []
    with p.open("r", newline="", encoding="utf-8") as f:
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
    p = argparse.ArgumentParser(description="Evaluate DR baseline under network constraints.")
    p.add_argument("session_dir", type=str, help="Path to data/raw_logs/<session_id>/")

    # DR config
    p.add_argument("--interval", type=int, default=10, help="Periodic update interval (tics).")
    p.add_argument("--pos-eps", type=float, default=5.0, help="Position threshold (units).")
    p.add_argument("--yaw-eps", type=float, default=5.0, help="Yaw threshold (deg).")
    p.add_argument("--pitch-eps", type=float, default=5.0, help="Pitch threshold (deg).")

    # Network config
    p.add_argument("--tps", type=float, default=35.0, help="Tics per second (default 35).")
    p.add_argument("--latency-ms", type=float, default=80.0, help="One-way latency (ms).")
    p.add_argument("--jitter-ms", type=float, default=20.0, help="Jitter uniform range (ms).")
    p.add_argument("--loss", type=float, default=0.01, help="Loss rate in [0,1].")
    p.add_argument(
        "--bandwidth-kbps", type=float, default=64.0, help="Bandwidth cap (kbps). 0=uncapped."
    )

    # Message model (kept configurable for Zandronum later)
    p.add_argument("--header-bytes", type=int, default=28, help="Header bytes per packet.")
    p.add_argument(
        "--payload-bytes",
        type=int,
        default=20,
        help="Payload bytes for state (default 5 floats*4).",
    )

    p.add_argument(
        "--seed", type=int, default=12345, help="Deterministic RNG seed for jitter/loss."
    )
    p.add_argument(
        "--out",
        type=str,
        default="evaluation_dr_net.json",
        help="Output filename written inside session_dir.",
    )

    args = p.parse_args(argv)
    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        print(f"FAIL: session_dir does not exist: {session_dir}", file=sys.stderr)
        return 2

    states = load_states(session_dir)

    dr_cfg = DRConfig(
        update_interval_tics=int(args.interval),
        pos_eps=float(args.pos_eps),
        yaw_eps_deg=float(args.yaw_eps),
        pitch_eps_deg=float(args.pitch_eps),
    )
    net_cfg = NetConfig(
        tics_per_second=float(args.tps),
        latency_ms=float(args.latency_ms),
        jitter_ms=float(args.jitter_ms),
        loss_rate=float(args.loss),
        bandwidth_kbps=float(args.bandwidth_kbps),
        header_bytes=int(args.header_bytes),
        payload_bytes_state=int(args.payload_bytes),
        rng_seed=int(args.seed),
    )

    summary = evaluate_dr_under_network(states, dr_cfg, net_cfg)

    payload = {
        "method": "dead_reckoning_net_v1",
        "dr_config": {
            "update_interval_tics": dr_cfg.update_interval_tics,
            "pos_eps": dr_cfg.pos_eps,
            "yaw_eps_deg": dr_cfg.yaw_eps_deg,
            "pitch_eps_deg": dr_cfg.pitch_eps_deg,
        },
        "net_config": {
            "tics_per_second": net_cfg.tics_per_second,
            "latency_ms": net_cfg.latency_ms,
            "jitter_ms": net_cfg.jitter_ms,
            "loss_rate": net_cfg.loss_rate,
            "bandwidth_kbps": net_cfg.bandwidth_kbps,
            "header_bytes": net_cfg.header_bytes,
            "payload_bytes_state": net_cfg.payload_bytes_state,
            "rng_seed": net_cfg.rng_seed,
        },
        "summary": {
            "n_tics": summary.n_tics,
            "dr_send_packets": summary.dr_send_packets,
            "dr_send_bytes": summary.dr_send_bytes,
            "baseline_send_packets": summary.baseline_send_packets,
            "baseline_send_bytes": summary.baseline_send_bytes,
            "savings_ratio_packets": summary.savings_ratio_packets,
            "savings_ratio_bytes": summary.savings_ratio_bytes,
            "net": {
                "sent_packets": summary.net.sent_packets,
                "delivered_packets": summary.net.delivered_packets,
                "dropped_packets": summary.net.dropped_packets,
                "sent_bytes": summary.net.sent_bytes,
                "delivered_bytes": summary.net.delivered_bytes,
                "avg_one_way_delay_ms": summary.net.avg_one_way_delay_ms,
                "max_queue_delay_ms": summary.net.max_queue_delay_ms,
            },
            "client_error": {
                "pos_mae": summary.client_error.pos_mae,
                "yaw_mae_deg": summary.client_error.yaw_mae_deg,
                "pitch_mae_deg": summary.client_error.pitch_mae_deg,
                "pos_max": summary.client_error.pos_max,
                "yaw_max_deg": summary.client_error.yaw_max_deg,
                "pitch_max_deg": summary.client_error.pitch_max_deg,
            },
        },
    }

    out_path = session_dir / str(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("OK")
    print("session_dir:", str(session_dir))
    print("out:", str(out_path))
    print("savings_ratio_bytes:", summary.savings_ratio_bytes)
    print("client_pos_mae:", summary.client_error.pos_mae)
    print("client_yaw_mae_deg:", summary.client_error.yaw_mae_deg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
