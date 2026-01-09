from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

NUMERIC_COLS = {
    "tics_per_second",
    "interval_tics",
    "pos_eps",
    "yaw_eps_deg",
    "pitch_eps_deg",
    "header_bytes",
    "payload_bytes_state",
    "rng_seed",
    "baseline_packets",
    "baseline_bytes",
    "dr_sent_packets",
    "dr_sent_bytes",
    "savings_ratio_packets",
    "savings_ratio_bytes",
    "delivered_packets",
    "dropped_packets",
    "avg_one_way_delay_ms",
    "max_queue_delay_ms",
    "client_pos_mae",
    "client_yaw_mae_deg",
    "client_pitch_mae_deg",
    "client_pos_max",
    "client_yaw_max_deg",
    "client_pitch_max_deg",
}


GROUP_SUMMARY_KEYS = ("scenario_name", "profile", "net_preset", "interval_tics")
BEST_KEYS = ("scenario_name", "profile", "net_preset")


@dataclass
class Row:
    data: dict[str, object]


def _to_num(s: str) -> float | int:
    # Prefer int if it looks integral; else float
    if s.strip() == "":
        return float("nan")
    try:
        if "." not in s and "e" not in s.lower():
            return int(s)
    except Exception:
        pass
    return float(s)


def read_csv_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")

    rows: list[Row] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV has no header")

        for raw in r:
            d: dict[str, object] = {}
            for k, v in raw.items():
                if k in NUMERIC_COLS:
                    d[k] = _to_num(v or "")
                else:
                    d[k] = v or ""
            rows.append(Row(d))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def mean(values: list[float]) -> float:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return float("nan")
    return sum(xs) / float(len(xs))


def group_summary(rows: list[Row]) -> list[dict[str, object]]:
    # Group by keys and compute mean of key metrics
    groups: dict[tuple[object, ...], list[Row]] = {}
    for r in rows:
        key = tuple(r.data.get(k) for k in GROUP_SUMMARY_KEYS)
        groups.setdefault(key, []).append(r)

    out: list[dict[str, object]] = []

    for key, rs in sorted(
        groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]), float(x[0][3]))
    ):
        sample = rs[0].data

        def col_f(name: str) -> list[float]:
            return [float(rr.data.get(name, float("nan"))) for rr in rs]

        row = {
            "scenario_name": sample.get("scenario_name", ""),
            "profile": sample.get("profile", ""),
            "net_preset": sample.get("net_preset", ""),
            "interval_tics": sample.get("interval_tics", float("nan")),
            "n_rows": len(rs),
            "mean_savings_ratio_bytes": mean(col_f("savings_ratio_bytes")),
            "mean_savings_ratio_packets": mean(col_f("savings_ratio_packets")),
            "mean_client_pos_mae": mean(col_f("client_pos_mae")),
            "mean_client_yaw_mae_deg": mean(col_f("client_yaw_mae_deg")),
            "mean_avg_one_way_delay_ms": mean(col_f("avg_one_way_delay_ms")),
            "mean_dropped_packets": mean(col_f("dropped_packets")),
        }
        out.append(row)

    return out


def pick_best(
    rows: list[Row],
    metric: str,
    pos_mae_max: float,
    yaw_mae_max: float,
) -> list[dict[str, object]]:
    """
    For each (scenario, profile, net_preset), pick the row with max(metric).
    Prefer rows that satisfy error constraints; if none satisfy, pick best overall
    and mark constraint_met=false.
    """
    groups: dict[tuple[object, ...], list[Row]] = {}
    for r in rows:
        key = tuple(r.data.get(k) for k in BEST_KEYS)
        groups.setdefault(key, []).append(r)

    best_rows: list[dict[str, object]] = []

    for key, rs in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), str(x[0][2]))):

        def metric_val(rr: Row) -> float:
            v = rr.data.get(metric, float("nan"))
            return float(v) if v is not None else float("nan")

        def meets(rr: Row) -> bool:
            p = float(rr.data.get("client_pos_mae", float("nan")))
            y = float(rr.data.get("client_yaw_mae_deg", float("nan")))
            return not math.isnan(p) and not math.isnan(y) and p <= pos_mae_max and y <= yaw_mae_max

        valid = [rr for rr in rs if meets(rr)]
        chosen_pool = valid if valid else rs

        # Choose max(metric)
        chosen = max(
            chosen_pool,
            key=lambda rr: (metric_val(rr), -float(rr.data.get("client_pos_mae", float("inf")))),
        )
        out = dict(chosen.data)  # copy all columns
        out["constraint_pos_mae_max"] = pos_mae_max
        out["constraint_yaw_mae_max"] = yaw_mae_max
        out["constraint_met"] = bool(valid)  # if we had at least one valid row in pool
        best_rows.append({k: out.get(k, "") for k in sorted(out.keys())})

    return best_rows


def try_make_plots(
    rows: list[Row],
    plots_dir: Path,
    metric: str,
) -> str | None:
    """
    Create a few PNG plots if matplotlib is installed.
    Returns None on success, or an error string if skipped/failed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return f"matplotlib not available ({type(e).__name__}: {e}); skipping plots."

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by (scenario_name, profile)
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        s = str(r.data.get("scenario_name", "unknown"))
        p = str(r.data.get("profile", "unknown"))
        groups.setdefault((s, p), []).append(r)

    # Scatter plots: savings vs pos MAE
    for (scenario, profile), rs in groups.items():
        xs = [float(rr.data.get("client_pos_mae", float("nan"))) for rr in rs]
        ys = [float(rr.data.get(metric, float("nan"))) for rr in rs]

        plt.figure()
        plt.scatter(xs, ys)
        plt.xlabel("client_pos_mae")
        plt.ylabel(metric)
        plt.title(f"{scenario} / {profile}: {metric} vs client_pos_mae")
        out = plots_dir / f"scatter_{scenario}_{profile}.png".replace(" ", "_")
        plt.savefig(out, dpi=150)
        plt.close()

    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report DR-under-network sweep results (tables + plots)."
    )
    ap.add_argument(
        "--in", dest="in_csv", type=str, default="results/dr_net_sweep.csv", help="Input sweep CSV."
    )
    ap.add_argument(
        "--summary-out",
        type=str,
        default="results/dr_net_summary.csv",
        help="Aggregated summary CSV.",
    )
    ap.add_argument(
        "--best-out", type=str, default="results/dr_net_best.csv", help="Best-per-group CSV."
    )
    ap.add_argument(
        "--plots-dir", type=str, default="results/plots", help="Directory for plots (PNG)."
    )

    ap.add_argument(
        "--metric",
        type=str,
        default="savings_ratio_bytes",
        choices=["savings_ratio_bytes", "savings_ratio_packets"],
        help="Optimization metric for best-per-group selection.",
    )
    ap.add_argument("--pos-mae-max", type=float, default=5.0, help="Constraint for client_pos_mae.")
    ap.add_argument(
        "--yaw-mae-max", type=float, default=5.0, help="Constraint for client_yaw_mae_deg."
    )

    args = ap.parse_args(argv)

    in_csv = Path(args.in_csv).resolve()
    summary_out = Path(args.summary_out).resolve()
    best_out = Path(args.best_out).resolve()
    plots_dir = Path(args.plots_dir).resolve()

    rows = read_csv_rows(in_csv)

    summary = group_summary(rows)
    summary_cols = list(summary[0].keys()) if summary else list(GROUP_SUMMARY_KEYS)

    write_csv(summary_out, summary_cols, summary)

    best = pick_best(rows, args.metric, args.pos_mae_max, args.yaw_mae_max)
    # For best rows, columns vary; take union and write consistently
    best_cols_set = set()
    for r in best:
        best_cols_set.update(r.keys())
    best_cols = sorted(best_cols_set)

    write_csv(best_out, best_cols, best)

    plot_status = try_make_plots(rows, plots_dir, args.metric)

    print("OK")
    print("in:", str(in_csv))
    print("summary_out:", str(summary_out))
    print("best_out:", str(best_out))
    if plot_status is None:
        print("plots:", str(plots_dir))
    else:
        print("plots: skipped:", plot_status)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
