from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Button bit mapping (your confirmed mapping)
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
class TrainConfig:
    epochs: int
    lr: float
    batch_size: int
    weight_decay: float
    seed: int
    train_frac: float
    val_frac: float
    use_action_multihot: bool
    normalize_inputs: bool
    normalize_targets: bool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _action_multihot(mask: np.ndarray) -> np.ndarray:
    # mask: shape (N,)
    out = np.zeros((mask.shape[0], len(BTN_ORDER)), dtype=np.float32)
    for j, name in enumerate(BTN_ORDER):
        bit = BTN[name]
        out[:, j] = ((mask & bit) != 0).astype(np.float32)
    return out


def _build_xy(df: pd.DataFrame, use_action_multihot: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised next-step regression:
      input_t = [x,y,z,yaw,pitch, action_features]
      target_t = [dx,dy,dz, dyaw, dpitch] where delta is (t+1 - t)
    """
    req = ["tic", "pos_x", "pos_y", "pos_z", "yaw_deg", "pitch_deg", "action_mask"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"states.csv missing columns: {missing}")

    df = df.sort_values("tic").reset_index(drop=True)

    # Build consecutive pairs (t -> t+1). We require strictly increasing tics.
    t = df["tic"].to_numpy()
    ok = t[1:] > t[:-1]
    idx = np.where(ok)[0]  # indices where pair (i, i+1) is valid

    s = df[["pos_x", "pos_y", "pos_z", "yaw_deg", "pitch_deg"]].to_numpy(dtype=np.float32)
    a = df["action_mask"].to_numpy(dtype=np.int64)

    x0 = s[idx]  # state at i
    x1 = s[idx + 1]  # state at i+1
    da = a[idx]  # action mask at i

    # Targets: delta in position + delta in angles (simple difference; acceptable for small steps)
    y = (x1 - x0).astype(np.float32)  # dx,dy,dz, dyaw, dpitch

    # Inputs
    if use_action_multihot:
        am = _action_multihot(da.astype(np.int64))
        X = np.concatenate([x0, am], axis=1).astype(np.float32)
    else:
        # Raw integer mask as a single scalar feature (less learnable; multi-hot is recommended)
        X = np.concatenate([x0, da.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)

    return X, y


def _split_time_series(X: np.ndarray, y: np.ndarray, train_frac: float, val_frac: float) -> tuple:
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_train = max(n_train, 1)
    n_val = max(n_val, 1) if n - n_train > 1 else 0
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    Xte, yte = X[n_train + n_val :], y[n_train + n_val :]
    return (Xtr, ytr, Xva, yva, Xte, yte)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return mu.astype(np.float32), sig.astype(np.float32)


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((x - mu) / sig).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train an MLP predictor on Zandronum states.csv.")
    ap.add_argument(
        "--session", type=Path, required=True, help="Session directory containing states.csv"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("models/mlp_zan_v1.pt"), help="Output checkpoint path"
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-action-multihot", action="store_true")
    ap.add_argument("--no-normalize-inputs", action="store_true")
    ap.add_argument("--no-normalize-targets", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        weight_decay=1e-4,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_action_multihot=not args.no_action_multihot,
        normalize_inputs=not args.no_normalize_inputs,
        normalize_targets=not args.no_normalize_targets,
    )

    _set_seed(cfg.seed)

    states_csv = args.session / "states.csv"
    if not states_csv.exists():
        raise SystemExit(f"Missing {states_csv}")

    df = pd.read_csv(states_csv)
    X, y = _build_xy(df, use_action_multihot=cfg.use_action_multihot)
    Xtr, ytr, Xva, yva, Xte, yte = _split_time_series(X, y, cfg.train_frac, cfg.val_frac)

    # Standardize
    x_mu = x_sig = y_mu = y_sig = None
    if cfg.normalize_inputs:
        x_mu, x_sig = _standardize_fit(Xtr)
        Xtr = _standardize_apply(Xtr, x_mu, x_sig)
        Xva = _standardize_apply(Xva, x_mu, x_sig) if len(Xva) else Xva
        Xte = _standardize_apply(Xte, x_mu, x_sig)
    if cfg.normalize_targets:
        y_mu, y_sig = _standardize_fit(ytr)
        ytr = _standardize_apply(ytr, y_mu, y_sig)
        yva = _standardize_apply(yva, y_mu, y_sig) if len(yva) else yva
        yte = _standardize_apply(yte, y_mu, y_sig)

    device = torch.device("cpu")
    in_dim = Xtr.shape[1]
    out_dim = ytr.shape[1]
    model = MLP(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    def to_t(xn: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(xn).to(device)

    # Logging
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Training loop
    n = Xtr.shape[0]
    for ep in range(cfg.epochs):
        model.train()
        perm = np.random.permutation(n)
        ep_loss = 0.0
        batches = 0

        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = to_t(Xtr[idx])
            yb = to_t(ytr[idx])

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())
            batches += 1

        tr = ep_loss / max(batches, 1)
        train_losses.append(tr)

        # Validation
        model.eval()
        if len(Xva):
            with torch.no_grad():
                pv = model(to_t(Xva))
                lv = loss_fn(pv, to_t(yva)).item()
            val_losses.append(float(lv))
        else:
            val_losses.append(float("nan"))

        print(
            f"epoch={ep+1:03d}/{cfg.epochs} train_mse={train_losses[-1]:.6f} val_mse={val_losses[-1]:.6f}"
        )

    # Save checkpoint with scalers + curves for plotting later
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "in_dim": in_dim,
        "hidden": args.hidden,
        "out_dim": out_dim,
        "cfg": asdict(cfg),
        "x_mu": x_mu,
        "x_sig": x_sig,
        "y_mu": y_mu,
        "y_sig": y_sig,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "btn_order": BTN_ORDER,
        "btn_map": BTN,
    }
    torch.save(ckpt, args.out)

    # Also write a small JSON beside it (human-readable)
    meta = {
        "session": str(args.session),
        "rows_total": int(len(df)),
        "pairs_used": int(X.shape[0]),
        "train_pairs": int(Xtr.shape[0]),
        "val_pairs": int(Xva.shape[0]),
        "test_pairs": int(Xte.shape[0]),
        "checkpoint": str(args.out),
        "cfg": asdict(cfg),
    }
    (args.out.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK")
    print("saved:", str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
