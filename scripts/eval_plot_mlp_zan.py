from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Must match your training script
BTN = {
    "ATTACK": 1,
    "USE": 2,
    "JUMP": 4,
    "SHIFT": 256,
    "D": 1024,
    "A": 2048,
    "S": 4096,
    "W": 8192,
}
BTN_ORDER = ["W", "A", "S", "D", "ATTACK", "USE", "JUMP", "SHIFT"]


@dataclass(frozen=True)
class State:
    tic: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


def _euclid3(dx: float, dy: float, dz: float) -> float:
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _wrap_deg(a: float) -> float:
    # Map angle into [-180, 180)
    x = (a + 180.0) % 360.0 - 180.0
    return float(x)


def _ang_abs_err_deg(a: float, b: float) -> float:
    return abs(_wrap_deg(a - b))


def _action_multihot(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], len(BTN_ORDER)), dtype=np.float32)
    for j, name in enumerate(BTN_ORDER):
        bit = BTN[name]
        out[:, j] = ((mask & bit) != 0).astype(np.float32)
    return out


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((x - mu) / sig).astype(np.float32)


def _unstandardize_apply(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (x * sig + mu).astype(np.float32)


def load_states_csv(session_dir: Path) -> tuple[list[State], list[int]]:
    p = session_dir / "states.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

    df = pd.read_csv(p).sort_values("tic").reset_index(drop=True)

    req = ["tic", "pos_x", "pos_y", "pos_z", "yaw_deg", "pitch_deg"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"states.csv missing column: {c}")

    if "action_mask" in df.columns:
        masks = df["action_mask"].to_numpy(dtype=np.int64).tolist()
    elif "mask" in df.columns:
        masks = df["mask"].to_numpy(dtype=np.int64).tolist()
    else:
        masks = [0 for _ in range(len(df))]

    states = [
        State(
            tic=int(r.tic),
            x=float(r.pos_x),
            y=float(r.pos_y),
            z=float(r.pos_z),
            yaw=float(r.yaw_deg),
            pitch=float(r.pitch_deg),
        )
        for r in df.itertuples(index=False)
    ]

    if len(states) < 3:
        raise ValueError("Need at least 3 rows to evaluate")

    return states, masks


def split_time(states: list[State], masks: list[int], train_frac: float, val_frac: float) -> tuple:
    n = len(states)
    n_train = max(int(n * train_frac), 2)
    n_val = max(int(n * val_frac), 0)
    if n_train + n_val >= n - 1:
        n_val = max(0, (n - 1) - n_train)

    # We evaluate on the tail after train+val
    i0 = n_train + n_val
    return (states[:n_train], masks[:n_train], states[i0:], masks[i0:])


def one_step_baselines(
    s_t: State,
    s_t1: State,
    s_tm1: State | None,
) -> tuple[State, State]:
    # Baseline 1: zero delta (predict no change)
    zero = State(tic=s_t1.tic, x=s_t.x, y=s_t.y, z=s_t.z, yaw=s_t.yaw, pitch=s_t.pitch)

    # Baseline 2: constant-velocity DR estimated from last two GT states
    if s_tm1 is None:
        dr = zero
    else:
        dt = max(1, s_t.tic - s_tm1.tic)
        vx = (s_t.x - s_tm1.x) / dt
        vy = (s_t.y - s_tm1.y) / dt
        vz = (s_t.z - s_tm1.z) / dt

        dyaw = _wrap_deg(s_t.yaw - s_tm1.yaw) / dt
        dpitch = _wrap_deg(s_t.pitch - s_tm1.pitch) / dt

        dt2 = max(1, s_t1.tic - s_t.tic)
        dr = State(
            tic=s_t1.tic,
            x=s_t.x + vx * dt2,
            y=s_t.y + vy * dt2,
            z=s_t.z + vz * dt2,
            yaw=s_t.yaw + dyaw * dt2,
            pitch=s_t.pitch + dpitch * dt2,
        )

    return zero, dr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate trained MLP on Zandronum trace and plot errors."
    )
    ap.add_argument(
        "--session", type=Path, required=True, help="Converted session dir containing states.csv"
    )
    ap.add_argument(
        "--ckpt", type=Path, required=True, help="Trained checkpoint, e.g. models/mlp_zan_v1.pt"
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/plots"))
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rollout", type=int, default=20, help="Closed-loop rollout horizon in tics")
    args = ap.parse_args()

    states, masks = load_states_csv(args.session)
    _, _, test_states, test_masks = split_time(states, masks, args.train_frac, args.val_frac)

    if len(test_states) < 3:
        raise SystemExit("Test tail too small; increase session length or adjust split fractions.")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    in_dim = int(ckpt["in_dim"])
    hidden = int(ckpt["hidden"])
    out_dim = int(ckpt["out_dim"])

    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_action_multihot = bool(ckpt["cfg"]["use_action_multihot"])
    norm_in = bool(ckpt["cfg"]["normalize_inputs"])
    norm_tgt = bool(ckpt["cfg"]["normalize_targets"])

    x_mu = ckpt.get("x_mu", None)
    x_sig = ckpt.get("x_sig", None)
    y_mu = ckpt.get("y_mu", None)
    y_sig = ckpt.get("y_sig", None)

    def featurize(s: State, mask: int) -> np.ndarray:
        base = np.array([s.x, s.y, s.z, s.yaw, s.pitch], dtype=np.float32).reshape(1, -1)
        if use_action_multihot:
            am = _action_multihot(np.array([mask], dtype=np.int64))
            x = np.concatenate([base, am], axis=1).astype(np.float32)
        else:
            x = np.concatenate([base, np.array([[float(mask)]], dtype=np.float32)], axis=1).astype(
                np.float32
            )

        if norm_in and x_mu is not None and x_sig is not None:
            x = _standardize_apply(x, x_mu, x_sig)
        return x

    def predict_delta(s: State, mask: int) -> np.ndarray:
        x = featurize(s, mask)
        with torch.no_grad():
            yhat = model(torch.from_numpy(x)).cpu().numpy().astype(np.float32)  # shape (1,5)
        if norm_tgt and y_mu is not None and y_sig is not None:
            yhat = _unstandardize_apply(yhat, y_mu, y_sig)
        return yhat.reshape(-1)  # (5,)

    # ------------------------------------------------------------
    # One-step errors over the test tail
    # ------------------------------------------------------------
    tics = []
    pos_err_mlp = []
    yaw_err_mlp = []
    pos_err_dr = []
    yaw_err_dr = []
    pos_err_zero = []
    yaw_err_zero = []

    # We need a GT previous state for DR estimate; use local indexing over full list.
    full_by_tic = {s.tic: s for s in states}

    for i in range(1, len(test_states)):
        s_t = test_states[i - 1]
        s_t1 = test_states[i]
        s_tm1 = full_by_tic.get(s_t.tic - 1, None)

        # Baselines
        zero, dr = one_step_baselines(s_t, s_t1, s_tm1)

        # MLP one-step prediction
        d = predict_delta(s_t, test_masks[i - 1])
        mlp = State(
            tic=s_t1.tic,
            x=s_t.x + float(d[0]),
            y=s_t.y + float(d[1]),
            z=s_t.z + float(d[2]),
            yaw=s_t.yaw + float(d[3]),
            pitch=s_t.pitch + float(d[4]),
        )

        # Errors
        tics.append(s_t1.tic)

        pos_err_zero.append(_euclid3(zero.x - s_t1.x, zero.y - s_t1.y, zero.z - s_t1.z))
        yaw_err_zero.append(_ang_abs_err_deg(zero.yaw, s_t1.yaw))

        pos_err_dr.append(_euclid3(dr.x - s_t1.x, dr.y - s_t1.y, dr.z - s_t1.z))
        yaw_err_dr.append(_ang_abs_err_deg(dr.yaw, s_t1.yaw))

        pos_err_mlp.append(_euclid3(mlp.x - s_t1.x, mlp.y - s_t1.y, mlp.z - s_t1.z))
        yaw_err_mlp.append(_ang_abs_err_deg(mlp.yaw, s_t1.yaw))

    # ------------------------------------------------------------
    # Closed-loop rollout error (harder, more realistic)
    # Roll forward K steps using model predictions only (no teacher forcing)
    # ------------------------------------------------------------
    K = max(int(args.rollout), 1)

    roll_tics = []
    roll_pos_err = []
    roll_yaw_err = []

    # Start at the beginning of the test tail
    cur = test_states[0]
    cur_tic = cur.tic

    # Roll for min(K, len(test_states)-1) steps, aligned to GT next tics
    steps = min(K, len(test_states) - 1)
    for j in range(steps):
        gt_next = test_states[j + 1]
        # Use action mask aligned with current tic if available
        am = test_masks[j] if j < len(test_masks) else 0

        d = predict_delta(cur, am)
        cur = State(
            tic=gt_next.tic,
            x=cur.x + float(d[0]),
            y=cur.y + float(d[1]),
            z=cur.z + float(d[2]),
            yaw=cur.yaw + float(d[3]),
            pitch=cur.pitch + float(d[4]),
        )
        cur_tic = cur.tic

        roll_tics.append(cur_tic)
        roll_pos_err.append(_euclid3(cur.x - gt_next.x, cur.y - gt_next.y, cur.z - gt_next.z))
        roll_yaw_err.append(_ang_abs_err_deg(cur.yaw, gt_next.yaw))

    # ------------------------------------------------------------
    # Write plots
    # ------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)

    # One-step position error
    plt.figure()
    plt.plot(tics, pos_err_zero, label="zero_delta")
    plt.plot(tics, pos_err_dr, label="dr_const_vel")
    plt.plot(tics, pos_err_mlp, label="mlp_one_step")
    plt.xlabel("tic")
    plt.ylabel("pos error (units)")
    plt.title("One-step position error on test tail")
    plt.legend()
    p1 = args.outdir / "zan_mlp_pos_error_one_step.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # One-step yaw error
    plt.figure()
    plt.plot(tics, yaw_err_zero, label="zero_delta")
    plt.plot(tics, yaw_err_dr, label="dr_const_vel")
    plt.plot(tics, yaw_err_mlp, label="mlp_one_step")
    plt.xlabel("tic")
    plt.ylabel("yaw abs error (deg)")
    plt.title("One-step yaw error on test tail")
    plt.legend()
    p2 = args.outdir / "zan_mlp_yaw_error_one_step.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    # Rollout errors
    plt.figure()
    plt.plot(roll_tics, roll_pos_err, label=f"mlp_rollout_{steps}t")
    plt.xlabel("tic")
    plt.ylabel("pos error (units)")
    plt.title("Closed-loop rollout position error (MLP)")
    plt.legend()
    p3 = args.outdir / "zan_mlp_pos_error_rollout.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(roll_tics, roll_yaw_err, label=f"mlp_rollout_{steps}t")
    plt.xlabel("tic")
    plt.ylabel("yaw abs error (deg)")
    plt.title("Closed-loop rollout yaw error (MLP)")
    plt.legend()
    p4 = args.outdir / "zan_mlp_yaw_error_rollout.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close()

    # Print quick numeric summary for confidence
    def _mean(x: list[float]) -> float:
        return float(np.mean(np.array(x, dtype=np.float32))) if x else 0.0

    print("OK")
    print(
        "test_one_step_pos_mean:",
        _mean(pos_err_mlp),
        "dr:",
        _mean(pos_err_dr),
        "zero:",
        _mean(pos_err_zero),
    )
    print(
        "test_one_step_yaw_mean:",
        _mean(yaw_err_mlp),
        "dr:",
        _mean(yaw_err_dr),
        "zero:",
        _mean(yaw_err_zero),
    )
    print("rollout_pos_mean:", _mean(roll_pos_err), "rollout_yaw_mean:", _mean(roll_yaw_err))
    print("wrote:")
    for p in (p1, p2, p3, p4):
        print(" ", str(p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
