from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Reuse the loader from your evaluate_dr_net.py to avoid schema drift.
from scripts.evaluate_dr_net import load_states  # noqa: E402
from src.eval.net_sim import (
    DRConfig,
    NetConfig,
    evaluate_predictor_under_network_factory,
)
from src.predictors.mlp_predictor import MLPPredictor


def iter_net_presets(seed: int) -> list[tuple[str, NetConfig]]:
    """
    Keep presets small and thesis-friendly.
    You can expand later once the pipeline is stable.
    """
    # Adjust tics_per_second if your Zandronum mapping changes later.
    tps = 35.0
    header = 28
    payload = 32

    presets: list[tuple[str, NetConfig]] = [
        (
            "low_latency",
            NetConfig(
                tics_per_second=tps,
                latency_ms=20.0,
                jitter_ms=3.0,
                loss_rate=0.0,
                bandwidth_kbps=0.0,
                header_bytes=header,
                payload_bytes_state=payload,
                rng_seed=seed,
            ),
        ),
        (
            "mid_latency",
            NetConfig(
                tics_per_second=tps,
                latency_ms=50.0,
                jitter_ms=5.0,
                loss_rate=0.01,
                bandwidth_kbps=0.0,
                header_bytes=header,
                payload_bytes_state=payload,
                rng_seed=seed,
            ),
        ),
        (
            "constrained",
            NetConfig(
                tics_per_second=tps,
                latency_ms=80.0,
                jitter_ms=10.0,
                loss_rate=0.03,
                bandwidth_kbps=256.0,
                header_bytes=header,
                payload_bytes_state=payload,
                rng_seed=seed,
            ),
        ),
    ]
    return presets


def iter_dr_grid() -> list[tuple[str, DRConfig]]:
    """
    A small grid (expand later).
    update_interval_tics is the "periodic send" interval.
    eps thresholds control correction sends.
    """
    grid: list[tuple[str, DRConfig]] = []
    for k in [5, 7, 10]:
        for pos_eps in [2.5, 5.0, 7.5]:
            for ang_eps in [3.0, 5.0, 8.0]:
                tag = f"k{k}_p{pos_eps:g}_a{ang_eps:g}"
                grid.append(
                    (
                        tag,
                        DRConfig(
                            update_interval_tics=k,
                            pos_eps=float(pos_eps),
                            yaw_eps_deg=float(ang_eps),
                            pitch_eps_deg=float(ang_eps),
                        ),
                    )
                )
    return grid


def write_row(w: csv.DictWriter, row: dict[str, Any]) -> None:
    w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep MLP predictor under network simulator.")
    ap.add_argument("--sessions-root", type=str, default="data/raw_logs")
    ap.add_argument("--glob", type=str, default="session_smoke_*")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ckpt", type=str, default="models/mlp_v1_w8_h1.pt")
    ap.add_argument("--out", type=str, default="results/mlp_net_sweep.csv")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_root).resolve()
    session_dirs = sorted([p for p in sessions_root.glob(args.glob) if p.is_dir()])
    if not session_dirs:
        raise SystemExit(f"No sessions found under {sessions_root} with glob {args.glob}")

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net_presets = iter_net_presets(seed=int(args.seed))
    dr_grid = iter_dr_grid()

    fieldnames = [
        "session_id",
        "net_preset",
        "dr_cfg_tag",
        "n_tics",
        "dr_send_packets",
        "dr_send_bytes",
        "baseline_send_packets",
        "baseline_send_bytes",
        "savings_ratio_packets",
        "savings_ratio_bytes",
        "pos_mae",
        "yaw_mae_deg",
        "pitch_mae_deg",
        "pos_max",
        "yaw_max_deg",
        "pitch_max_deg",
        "net_sent_packets",
        "net_delivered_packets",
        "net_dropped_packets",
        "net_sent_bytes",
        "net_delivered_bytes",
        "net_avg_one_way_delay_ms",
        "net_max_queue_delay_ms",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for sd in session_dirs:
            session_id = sd.name
            states_path = sd / "states.csv"
            if not states_path.exists():
                print(f"SKIP (missing states.csv): {session_id}")
                continue
            try:
                states, action_masks = load_states(sd)
            except (FileNotFoundError, ValueError) as e:
                print(f"SKIP (bad session): {session_id} ({type(e).__name__}: {e})")
                continue

            for net_name, net_cfg in net_presets:
                for dr_tag, dr_cfg in dr_grid:
                    summary = evaluate_predictor_under_network_factory(
                        states=states,
                        action_masks=action_masks,
                        predictor_factory=lambda p=ckpt_path: MLPPredictor(ckpt_path=p),
                        dr_cfg=dr_cfg,
                        net_cfg=net_cfg,
                    )

                    row = {
                        "session_id": session_id,
                        "net_preset": net_name,
                        "dr_cfg_tag": dr_tag,
                        "n_tics": summary.n_tics,
                        "dr_send_packets": summary.dr_send_packets,
                        "dr_send_bytes": summary.dr_send_bytes,
                        "baseline_send_packets": summary.baseline_send_packets,
                        "baseline_send_bytes": summary.baseline_send_bytes,
                        "savings_ratio_packets": summary.savings_ratio_packets,
                        "savings_ratio_bytes": summary.savings_ratio_bytes,
                        "pos_mae": summary.client_error.pos_mae,
                        "yaw_mae_deg": summary.client_error.yaw_mae_deg,
                        "pitch_mae_deg": summary.client_error.pitch_mae_deg,
                        "pos_max": summary.client_error.pos_max,
                        "yaw_max_deg": summary.client_error.yaw_max_deg,
                        "pitch_max_deg": summary.client_error.pitch_max_deg,
                        "net_sent_packets": summary.net.sent_packets,
                        "net_delivered_packets": summary.net.delivered_packets,
                        "net_dropped_packets": summary.net.dropped_packets,
                        "net_sent_bytes": summary.net.sent_bytes,
                        "net_delivered_bytes": summary.net.delivered_bytes,
                        "net_avg_one_way_delay_ms": summary.net.avg_one_way_delay_ms,
                        "net_max_queue_delay_ms": summary.net.max_queue_delay_ms,
                    }
                    write_row(w, row)

    meta = {
        "sessions_root": str(sessions_root),
        "glob": args.glob,
        "seed": int(args.seed),
        "ckpt": str(ckpt_path),
        "net_presets": [n for n, _ in net_presets],
        "dr_grid_size": len(dr_grid),
        "sessions_found": len(session_dirs),
        "rows_written": len(session_dirs) * len(net_presets) * len(dr_grid),
        "out": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK")
    print("sessions_root:", str(sessions_root))
    print("glob:", args.glob)
    print("sessions_found:", len(session_dirs))
    print("rows_written:", meta["rows_written"])
    print("out:", str(out_path))
    print("meta:", str(meta_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
