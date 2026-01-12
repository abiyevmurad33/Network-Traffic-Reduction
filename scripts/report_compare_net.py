from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing {kind} column. Tried {candidates}. Available: {df.columns.tolist()}")


def _best_by_preset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick best row per (net_preset) by:
      1) maximize savings_ratio_bytes
      2) then minimize pos_mae
    """
    df2 = df.copy()
    df2 = df2.sort_values(
        by=["net_preset", "savings_ratio_bytes", "pos_mae"],
        ascending=[True, False, True],
    )
    return df2.groupby(["net_preset"], as_index=False).head(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare DR vs MLP best configs per net preset.")
    ap.add_argument("--dr", type=str, default="results/dr_net_sweep.csv")
    ap.add_argument("--mlp", type=str, default="results/mlp_net_sweep.csv")
    ap.add_argument("--out-summary", type=str, default="results/compare_net_summary.csv")
    ap.add_argument("--out-best", type=str, default="results/compare_net_best.csv")
    args = ap.parse_args()

    dr_path = Path(args.dr).resolve()
    mlp_path = Path(args.mlp).resolve()
    if not dr_path.exists():
        raise SystemExit(f"Missing: {dr_path}")
    if not mlp_path.exists():
        raise SystemExit(f"Missing: {mlp_path}")

    dr = pd.read_csv(dr_path)
    mlp = pd.read_csv(mlp_path)

    # Normalize columns if needed (assumes sweep_mlp matches these names)
    dr["method"] = "DR"
    mlp["method"] = "MLP"

    # Detect column names (DR/MLP sweeps may differ)
    net_col_dr = _pick_col(dr, ["net_preset", "net_profile", "net_name"], "net preset")
    net_col_mlp = _pick_col(mlp, ["net_preset", "net_profile", "net_name"], "net preset")

    # DR sweep may not have a single tag column; it may expose config as separate columns.
    cfg_col_dr = None
    for cand in ["dr_cfg_tag", "cfg_tag", "dr_cfg", "cfg", "config_tag"]:
        if cand in dr.columns:
            cfg_col_dr = cand
            break
    if cfg_col_dr is None:
        if cfg_col_dr is None:
            needed = ["interval_tics", "pos_eps", "yaw_eps_deg", "pitch_eps_deg"]
            missing = [c for c in needed if c not in dr.columns]
            if missing:
                raise KeyError(f"DR sweep missing config columns: {missing}")
            dr = dr.copy()
            dr["cfg_tag"] = (
                "k"
                + dr["interval_tics"].astype(str)
                + "_p"
                + dr["pos_eps"].astype(str)
                + "_y"
                + dr["yaw_eps_deg"].astype(str)
                + "_pi"
                + dr["pitch_eps_deg"].astype(str)
            )
        else:
            dr = dr.rename(columns={cfg_col_dr: "cfg_tag"})

    cfg_col_mlp = _pick_col(
        mlp,
        ["dr_cfg_tag", "cfg_tag", "dr_cfg", "cfg", "config_tag"],
        "config tag",
    )

    # Normalize net preset column
    dr = dr.rename(columns={net_col_dr: "net_preset"})
    mlp = mlp.rename(columns={net_col_mlp: "net_preset"})

    # Normalize metric column names to a common schema
    dr = dr.rename(
        columns={
            "baseline_packets": "baseline_send_packets",
            "baseline_bytes": "baseline_send_bytes",
            "dr_sent_packets": "dr_send_packets",
            "dr_sent_bytes": "dr_send_bytes",
            "delivered_packets": "net_delivered_packets",
            "dropped_packets": "net_dropped_packets",
            "client_pos_mae": "pos_mae",
            "client_yaw_mae_deg": "yaw_mae_deg",
            "client_pitch_mae_deg": "pitch_mae_deg",
            "client_pos_max": "pos_max",
            "client_yaw_max_deg": "yaw_max_deg",
            "client_pitch_max_deg": "pitch_max_deg",
        }
    )

    # Normalize MLP config tag column
    mlp = mlp.rename(columns={cfg_col_mlp: "cfg_tag"})

    def agg_mean(df: pd.DataFrame) -> pd.DataFrame:
        gcols = ["method", "net_preset", "cfg_tag"]
        num = df.select_dtypes(include="number").columns.tolist()
        return df.groupby(gcols, as_index=False)[num].mean()

    dr_m = agg_mean(dr)
    mlp_m = agg_mean(mlp)

    dr_best = _best_by_preset(dr_m)
    mlp_best = _best_by_preset(mlp_m)

    best = pd.concat([dr_best, mlp_best], ignore_index=True)
    best_out = Path(args.out_best).resolve()
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_out, index=False)

    # Produce a compact side-by-side summary
    pivot_cols = [
        "savings_ratio_bytes",
        "pos_mae",
        "yaw_mae_deg",
        "pitch_mae_deg",
        "pos_max",
        "yaw_max_deg",
        "pitch_max_deg",
        "dr_send_bytes",
        "baseline_send_bytes",
    ]
    summary = best.pivot(index="net_preset", columns="method", values=pivot_cols)
    # Flatten columns for CSV
    summary.columns = [f"{metric}_{method}" for metric, method in summary.columns]
    summary = summary.reset_index()

    summary_out = Path(args.out_summary).resolve()
    summary.to_csv(summary_out, index=False)

    print("OK")
    print("best_out:", str(best_out))
    print("summary_out:", str(summary_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
