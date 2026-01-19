from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize sweep CSV into compact tables (no plots).")
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        required=True,
        help="Input CSV (e.g., results/zandronum/dr_net_sweep_zan.csv)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output CSV summary path")
    ap.add_argument(
        "--group",
        type=str,
        default="net_preset,interval_tics,pos_eps,yaw_eps_deg,pitch_eps_deg",
        help="Comma-separated group-by columns",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    gcols = [c.strip() for c in args.group.split(",") if c.strip()]
    for c in gcols:
        if c not in df.columns:
            raise SystemExit(f"Missing group col {c}. Available: {df.columns.tolist()}")

    # Numeric columns we care about for Traffic Adapter v0
    num_cols = [
        "savings_ratio_packets",
        "savings_ratio_bytes",
        "client_pos_mae",
        "client_yaw_mae_deg",
        "client_pitch_mae_deg",
        "client_pos_max",
        "client_yaw_max_deg",
        "client_pitch_max_deg",
        "dropped_packets",
        "delivered_packets",
        "avg_one_way_delay_ms",
        "max_queue_delay_ms",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    agg = df.groupby(gcols, as_index=False)[num_cols].agg(["mean", "median", "min", "max"])
    # Flatten multiindex columns
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg.to_csv(args.out, index=False)

    print("OK")
    print("in:", str(args.inp))
    print("out:", str(args.out))
    print("rows_in:", len(df), "rows_out:", len(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
