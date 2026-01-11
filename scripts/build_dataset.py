from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eval.net_sim import State
from src.utils.angles import circular_diff_deg


@dataclass(frozen=True)
class Trace:
    session_id: str
    states: list[State]
    action_masks: list[int]


def load_states(session_dir: Path) -> tuple[list[State], list[int]]:
    p = session_dir / "states.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

    out: list[State] = []
    masks: list[int] = []

    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)

        mask_col = None
        if r.fieldnames is not None:
            if "action_mask" in r.fieldnames:
                mask_col = "action_mask"
            elif "mask" in r.fieldnames:
                mask_col = "mask"

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
            masks.append(int(row[mask_col]) if mask_col is not None else 0)

    if len(out) < 2:
        raise ValueError(f"Need at least 2 rows in {p}")
    if len(out) != len(masks):
        raise RuntimeError("states/masks length mismatch")

    return out, masks


def compute_deltas(states: list[State]) -> np.ndarray:
    """
    Returns array shape (T, 5):
      [dx, dy, dz, dyaw_deg, dpitch_deg]
    where row t corresponds to delta from states[t] -> states[t+1].
    So length is (len(states)-1).
    """
    T = len(states) - 1
    d = np.zeros((T, 5), dtype=np.float32)
    for i in range(T):
        a = states[i]
        b = states[i + 1]
        d[i, 0] = b.x - a.x
        d[i, 1] = b.y - a.y
        d[i, 2] = b.z - a.z
        d[i, 3] = float(circular_diff_deg(b.yaw, a.yaw))
        d[i, 4] = float(circular_diff_deg(b.pitch, a.pitch))
    return d


def build_xy(trace: Trace, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised samples from a single trace.

    We define:
      deltas[t] = state[t] -> state[t+1]  (t = 0..T-2)
      masks[t]  = action applied at state[t] (same indexing as states)

    Input sample at index k uses deltas[k-window .. k-1] and masks[k-window .. k-1]
    Target is deltas[k + horizon - 1]  (i.e., horizon=1 => deltas[k])

    Returns:
      X: (N, window*6)  where per-step features are [dx,dy,dz,dyaw,dpitch,action_mask]
      Y: (N, 5)         next delta vector
    """
    deltas = compute_deltas(trace.states)  # (T-1, 5)
    masks = np.array(trace.action_masks[:-1], dtype=np.int32)  # align to deltas length

    if deltas.shape[0] != masks.shape[0]:
        raise RuntimeError("deltas/masks alignment mismatch")

    T = deltas.shape[0]
    start = window
    end = T - (horizon - 1)
    if end <= start:
        return np.zeros((0, window * 6), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for k in range(start, end):
        hist_d = deltas[k - window : k, :]  # (window, 5)
        hist_m = masks[k - window : k].astype(np.float32).reshape(window, 1)
        hist = np.concatenate([hist_d, hist_m], axis=1)  # (window, 6)

        y_idx = k + horizon - 1
        y = deltas[y_idx, :]  # (5,)

        xs.append(hist.reshape(-1))
        ys.append(y)

    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y


def main() -> int:
    ap = argparse.ArgumentParser(description="Build dataset for MLP predictor from session traces.")
    ap.add_argument("--sessions-root", type=str, default="data/raw_logs")
    ap.add_argument("--glob", type=str, default="session_smoke_*")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="data/processed/datasets/v1_w8_h1_trainval.npz")

    args = ap.parse_args()

    root = Path(args.sessions_root).resolve()
    session_dirs = sorted([p for p in root.glob(args.glob) if p.is_dir()])
    if not session_dirs:
        raise SystemExit(f"No sessions found under {root} with glob {args.glob}")

    traces: list[Trace] = []
    for sd in session_dirs:
        session_id = sd.name
        states, masks = load_states(sd)
        traces.append(Trace(session_id=session_id, states=states, action_masks=masks))

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(traces))
    rng.shuffle(idxs)

    n_val = int(round(len(traces) * float(args.val_frac)))
    val_ids = set(idxs[:n_val].tolist())
    train_traces = [traces[i] for i in range(len(traces)) if i not in val_ids]
    val_traces = [traces[i] for i in range(len(traces)) if i in val_ids]

    X_tr, Y_tr = [], []
    X_va, Y_va = [], []

    for tr in train_traces:
        X, Y = build_xy(tr, window=args.window, horizon=args.horizon)
        if X.shape[0] > 0:
            X_tr.append(X)
            Y_tr.append(Y)

    for tr in val_traces:
        X, Y = build_xy(tr, window=args.window, horizon=args.horizon)
        if X.shape[0] > 0:
            X_va.append(X)
            Y_va.append(Y)

    if not X_tr:
        raise SystemExit("No training samples produced. Increase data or reduce window/horizon.")

    X_train = np.concatenate(X_tr, axis=0)
    Y_train = np.concatenate(Y_tr, axis=0)
    X_val = (
        np.concatenate(X_va, axis=0) if X_va else np.zeros((0, X_train.shape[1]), dtype=np.float32)
    )
    Y_val = (
        np.concatenate(Y_va, axis=0) if Y_va else np.zeros((0, Y_train.shape[1]), dtype=np.float32)
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    np.savez_compressed(
        out_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        feature_dim=int(X_train.shape[1]),
        target_dim=int(Y_train.shape[1]),
        window=int(args.window),
        horizon=int(args.horizon),
        seed=int(args.seed),
    )

    # Save metadata JSON
    meta = {
        "sessions_root": str(root),
        "glob": args.glob,
        "window": int(args.window),
        "horizon": int(args.horizon),
        "val_frac": float(args.val_frac),
        "seed": int(args.seed),
        "n_sessions_total": int(len(traces)),
        "n_sessions_train": int(len(train_traces)),
        "n_sessions_val": int(len(val_traces)),
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_val": int(X_val.shape[0]),
        "x_dim": int(X_train.shape[1]),
        "y_dim": int(Y_train.shape[1]),
        "x_layout": "window * [dx,dy,dz,dyaw_deg,dpitch_deg,action_mask]",
        "y_layout": "[dx,dy,dz,dyaw_deg,dpitch_deg] at horizon",
        "train_session_ids": [t.session_id for t in train_traces],
        "val_session_ids": [t.session_id for t in val_traces],
    }

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK")
    print("out:", str(out_path))
    print("meta:", str(meta_path))
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val:", X_val.shape, "Y_val:", Y_val.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
