from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def pareto_frontier(df: pd.DataFrame, x: str, y: str, x_maximize: bool = True) -> pd.DataFrame:
    """
    Return Pareto-efficient rows for:
      maximize x (savings) and minimize y (error).
    """
    d = df[[x, y, "method", "net_preset", "cfg_tag"]].dropna().copy()

    # Sort by x desc, then y asc
    d = d.sort_values(by=[x, y], ascending=[not x_maximize, True])

    best_y = float("inf")
    keep = []
    for _, r in d.iterrows():
        yy = float(r[y])
        if yy < best_y:
            keep.append(True)
            best_y = yy
        else:
            keep.append(False)
    out = d.loc[keep].copy()
    # For plotting a curve, sort by x ascending (left->right)
    out = out.sort_values(by=[x], ascending=True)
    return out


def agg_mean(df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["method", "net_preset", "cfg_tag"]
    num = df.select_dtypes(include="number").columns.tolist()
    return df.groupby(gcols, as_index=False)[num].mean()


def plot_one(
    df: pd.DataFrame,
    net_preset: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> None:
    d = df[df["net_preset"] == net_preset].copy()
    if d.empty:
        return

    plt.figure()
    for method in ["DR", "MLP"]:
        dd = d[d["method"] == method]
        if dd.empty:
            continue

        plt.scatter(dd["savings_ratio_bytes"], dd[ycol], label=f"{method} points")

        front = pareto_frontier(dd, x="savings_ratio_bytes", y=ycol, x_maximize=True)
        if not front.empty:
            plt.plot(front["savings_ratio_bytes"], front[ycol], label=f"{method} Pareto")

    plt.xlabel("Byte savings ratio (higher is better)")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot DR vs MLP Pareto comparisons per net preset.")
    ap.add_argument("--dr", type=str, default="results/dr_net_sweep.csv")
    ap.add_argument("--mlp", type=str, default="results/mlp_net_sweep.csv")
    ap.add_argument("--outdir", type=str, default="results/plots_compare")
    args = ap.parse_args()

    dr_path = Path(args.dr).resolve()
    mlp_path = Path(args.mlp).resolve()
    if not dr_path.exists():
        raise SystemExit(f"Missing: {dr_path}")
    if not mlp_path.exists():
        raise SystemExit(f"Missing: {mlp_path}")

    dr = pd.read_csv(dr_path)
    mlp = pd.read_csv(mlp_path)

    dr["method"] = "DR"
    mlp["method"] = "MLP"

    # Normalize DR columns to match MLP sweep schema
    dr = dr.rename(
        columns={
            "baseline_packets": "baseline_send_packets",
            "baseline_bytes": "baseline_send_bytes",
            "dr_sent_packets": "dr_send_packets",
            "dr_sent_bytes": "dr_send_bytes",
            "client_pos_mae": "pos_mae",
            "client_yaw_mae_deg": "yaw_mae_deg",
            "client_pitch_mae_deg": "pitch_mae_deg",
            "client_pos_max": "pos_max",
            "client_yaw_max_deg": "yaw_max_deg",
            "client_pitch_max_deg": "pitch_max_deg",
        }
    )

    # Build cfg_tag for DR
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

    # MLP already has dr_cfg_tag from our sweep script
    if "cfg_tag" not in mlp.columns:
        if "dr_cfg_tag" in mlp.columns:
            mlp = mlp.rename(columns={"dr_cfg_tag": "cfg_tag"})
        else:
            raise KeyError("MLP sweep missing cfg tag column (expected cfg_tag or dr_cfg_tag).")

    # Aggregate by config to stabilize
    all_df = pd.concat([dr, mlp], ignore_index=True)
    m = agg_mean(all_df)

    outdir = Path(args.outdir).resolve()
    presets = sorted(m["net_preset"].dropna().unique().tolist())

    for preset in presets:
        plot_one(
            m,
            net_preset=preset,
            ycol="pos_mae",
            out_path=outdir / f"pareto_{preset}_pos.png",
            title=f"DR vs MLP Pareto: {preset} (pos_mae)",
        )
        plot_one(
            m,
            net_preset=preset,
            ycol="yaw_mae_deg",
            out_path=outdir / f"pareto_{preset}_yaw.png",
            title=f"DR vs MLP Pareto: {preset} (yaw_mae_deg)",
        )

    print("OK")
    print("outdir:", str(outdir))
    print("presets:", presets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
