from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.eval.net_sim import DRConfig, NetConfig, State, evaluate_dr_under_network


@dataclass(frozen=True)
class NetPreset:
    name: str
    latency_ms: float
    jitter_ms: float
    loss_rate: float
    bandwidth_kbps: float


# Reasonable, thesis-friendly presets (one-way latency model)
NET_PRESETS: list[NetPreset] = [
    NetPreset("lan", latency_ms=10.0, jitter_ms=1.0, loss_rate=0.000, bandwidth_kbps=10_000.0),
    NetPreset("wifi_good", latency_ms=30.0, jitter_ms=5.0, loss_rate=0.002, bandwidth_kbps=5_000.0),
    NetPreset(
        "wifi_busy", latency_ms=60.0, jitter_ms=15.0, loss_rate=0.010, bandwidth_kbps=1_000.0
    ),
    NetPreset("mobile_ok", latency_ms=90.0, jitter_ms=25.0, loss_rate=0.020, bandwidth_kbps=256.0),
    NetPreset(
        "mobile_bad", latency_ms=140.0, jitter_ms=40.0, loss_rate=0.050, bandwidth_kbps=128.0
    ),
]


CSV_COLUMNS = [
    # session identification
    "session_id",
    "scenario_name",
    "profile",
    "tics_per_second",
    # sweep knobs
    "net_preset",
    "interval_tics",
    "pos_eps",
    "yaw_eps_deg",
    "pitch_eps_deg",
    "header_bytes",
    "payload_bytes_state",
    "rng_seed",
    # savings
    "baseline_packets",
    "baseline_bytes",
    "dr_sent_packets",
    "dr_sent_bytes",
    "savings_ratio_packets",
    "savings_ratio_bytes",
    # network delivery stats
    "delivered_packets",
    "dropped_packets",
    "avg_one_way_delay_ms",
    "max_queue_delay_ms",
    # client error under network
    "client_pos_mae",
    "client_yaw_mae_deg",
    "client_pitch_mae_deg",
    "client_pos_max",
    "client_yaw_max_deg",
    "client_pitch_max_deg",
]


def load_session_meta(session_dir: Path) -> dict:
    p = session_dir / "session.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_profile(meta: dict) -> str:
    # stored as players[0].notes = "profile=..."
    try:
        notes = meta["players"][0].get("notes", "")
        if isinstance(notes, str) and "profile=" in notes:
            # e.g., "profile=strafe_turn"
            parts = notes.split("profile=", 1)[1].strip()
            return parts
    except Exception:
        pass
    return "unknown"


def load_states(session_dir: Path) -> list[State]:
    p = session_dir / "states.csv"
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
        raise ValueError(f"Need at least 2 rows in {p}")
    return out


def iter_session_dirs(root: Path, glob_pattern: str) -> Iterable[Path]:
    # Only directories that contain both required files
    for d in sorted(root.glob(glob_pattern)):
        if d.is_dir() and (d / "states.csv").exists() and (d / "session.json").exists():
            yield d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep DR-under-network evaluation across multiple sessions."
    )
    ap.add_argument(
        "--sessions-root",
        type=str,
        default=str(Path("data") / "raw_logs"),
        help="Root folder containing session directories (default: data/raw_logs).",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="session_*",
        help="Glob pattern under sessions-root (default: session_*).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "dr_net_sweep.csv"),
        help="Output CSV path (default: results/dr_net_sweep.csv).",
    )

    # Sweep knobs
    ap.add_argument(
        "--intervals", type=str, default="5,10,20", help="Comma-separated intervals in tics."
    )
    ap.add_argument(
        "--pos-eps", type=str, default="2,5", help="Comma-separated position thresholds."
    )
    ap.add_argument("--yaw-eps", type=str, default="3,5", help="Comma-separated yaw thresholds.")
    ap.add_argument(
        "--pitch-eps", type=str, default="3,5", help="Comma-separated pitch thresholds."
    )

    # Message model (keep configurable; calibrate later for Zandronum)
    ap.add_argument("--header-bytes", type=int, default=28, help="Header bytes per packet.")
    ap.add_argument("--payload-bytes", type=int, default=20, help="Payload bytes for pose state.")
    ap.add_argument(
        "--seed", type=int, default=12345, help="RNG seed used for jitter/loss (deterministic)."
    )

    args = ap.parse_args(argv)

    sessions_root = Path(args.sessions_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    pos_eps_list = [float(x.strip()) for x in args.pos_eps.split(",") if x.strip()]
    yaw_eps_list = [float(x.strip()) for x in args.yaw_eps.split(",") if x.strip()]
    pitch_eps_list = [float(x.strip()) for x in args.pitch_eps.split(",") if x.strip()]

    session_dirs = list(iter_session_dirs(sessions_root, args.glob))
    if not session_dirs:
        raise SystemExit(f"No sessions found under {sessions_root} with glob={args.glob!r}")

    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()

        for sd in session_dirs:
            meta = load_session_meta(sd)
            session_id = meta.get("session_id", sd.name)
            scenario_name = meta.get("scenario_name", "unknown")
            profile = parse_profile(meta)

            # Prefer session.json tps if present, else default 35
            tps = float(meta.get("tics_per_second", 35.0))

            states = load_states(sd)

            for preset in NET_PRESETS:
                for interval in intervals:
                    for pos_eps in pos_eps_list:
                        for yaw_eps in yaw_eps_list:
                            for pitch_eps in pitch_eps_list:
                                dr_cfg = DRConfig(
                                    update_interval_tics=interval,
                                    pos_eps=pos_eps,
                                    yaw_eps_deg=yaw_eps,
                                    pitch_eps_deg=pitch_eps,
                                )
                                net_cfg = NetConfig(
                                    tics_per_second=tps,
                                    latency_ms=preset.latency_ms,
                                    jitter_ms=preset.jitter_ms,
                                    loss_rate=preset.loss_rate,
                                    bandwidth_kbps=preset.bandwidth_kbps,
                                    header_bytes=int(args.header_bytes),
                                    payload_bytes_state=int(args.payload_bytes),
                                    rng_seed=int(args.seed),
                                )

                                summary = evaluate_dr_under_network(states, dr_cfg, net_cfg)

                                w.writerow(
                                    {
                                        "session_id": session_id,
                                        "scenario_name": scenario_name,
                                        "profile": profile,
                                        "tics_per_second": tps,
                                        "net_preset": preset.name,
                                        "interval_tics": interval,
                                        "pos_eps": pos_eps,
                                        "yaw_eps_deg": yaw_eps,
                                        "pitch_eps_deg": pitch_eps,
                                        "header_bytes": int(args.header_bytes),
                                        "payload_bytes_state": int(args.payload_bytes),
                                        "rng_seed": int(args.seed),
                                        "baseline_packets": summary.baseline_send_packets,
                                        "baseline_bytes": summary.baseline_send_bytes,
                                        "dr_sent_packets": summary.dr_send_packets,
                                        "dr_sent_bytes": summary.dr_send_bytes,
                                        "savings_ratio_packets": summary.savings_ratio_packets,
                                        "savings_ratio_bytes": summary.savings_ratio_bytes,
                                        "delivered_packets": summary.net.delivered_packets,
                                        "dropped_packets": summary.net.dropped_packets,
                                        "avg_one_way_delay_ms": summary.net.avg_one_way_delay_ms,
                                        "max_queue_delay_ms": summary.net.max_queue_delay_ms,
                                        "client_pos_mae": summary.client_error.pos_mae,
                                        "client_yaw_mae_deg": summary.client_error.yaw_mae_deg,
                                        "client_pitch_mae_deg": summary.client_error.pitch_mae_deg,
                                        "client_pos_max": summary.client_error.pos_max,
                                        "client_yaw_max_deg": summary.client_error.yaw_max_deg,
                                        "client_pitch_max_deg": summary.client_error.pitch_max_deg,
                                    }
                                )
                                rows_written += 1

    print("OK")
    print("sessions_root:", str(sessions_root))
    print("glob:", args.glob)
    print("sessions_found:", len(session_dirs))
    print("rows_written:", rows_written)
    print("out:", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
