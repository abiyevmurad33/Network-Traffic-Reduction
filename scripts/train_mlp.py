from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP
from src.predictors.mlp_schema import FEATURE_SPEC_V1, TARGET_SPEC_V1


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="data/processed/datasets/v1_w8_h1_trainval.npz",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)

    # IMPORTANT: produce a "contracted" checkpoint that inference/sweeps can rely on.
    # Use a new name to avoid confusion with legacy checkpoints.
    ap.add_argument("--out", type=str, default="models/mlp_zan_v2.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}")

    d = np.load(data_path)
    X_train = d["X_train"].astype(np.float32)
    Y_train = d["Y_train"].astype(np.float32)
    X_val = d["X_val"].astype(np.float32)
    Y_val = d["Y_val"].astype(np.float32)

    # Sanity checks: fail fast if dataset doesn't match declared schema.
    # This stops the "patching cycle" caused by silent shape mismatches.
    if X_train.ndim != 2 or Y_train.ndim != 2:
        raise ValueError("Expected X_train/Y_train to be 2D arrays.")
    feature_dim = int(X_train.shape[1])
    target_dim = int(Y_train.shape[1])

    # Expect our target to remain 5-dim (dx,dy,dz,dyaw,dpitch)
    if target_dim != TARGET_SPEC_V1.dim:
        raise ValueError(
            f"Target dim mismatch: dataset has {target_dim} but schema expects {TARGET_SPEC_V1.dim}"
        )

    # Infer window size if the layout is (window*5 + 8 action bits)
    # This matches v1_w8_h1 datasets where features are 5*k + 8.
    window = None
    if feature_dim >= 13 and (feature_dim - 8) % 5 == 0:
        window = (feature_dim - 8) // 5
    if Y_train.shape[1] != TARGET_SPEC_V1.dim:
        raise ValueError(
            f"Target dim mismatch: dataset has {Y_train.shape[1]} but schema expects {TARGET_SPEC_V1.dim}"
        )

    x_mean, x_std = _standardize_fit(X_train)
    y_mean, y_std = _standardize_fit(Y_train)

    X_train_n = _standardize_apply(X_train, x_mean, x_std)
    Y_train_n = _standardize_apply(Y_train, y_mean, y_std)

    X_val_n = _standardize_apply(X_val, x_mean, x_std) if X_val.size else X_val.astype(np.float32)
    Y_val_n = _standardize_apply(Y_val, y_mean, y_std) if Y_val.size else Y_val.astype(np.float32)

    device = torch.device("cpu")
    model = MLP(in_dim=int(X_train.shape[1]), out_dim=int(Y_train.shape[1]), hidden=args.hidden).to(
        device
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(Y_train_n),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_dl = None
    if X_val_n.size:
        val_ds = TensorDataset(
            torch.from_numpy(X_val_n),
            torch.from_numpy(Y_val_n),
        )
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])

        tr_loss /= max(tr_n, 1)

        val_loss = float("nan")
        if val_dl is not None:
            model.eval()
            va_loss = 0.0
            va_n = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    va_loss += float(loss.item()) * int(xb.shape[0])
                    va_n += int(xb.shape[0])
            val_loss = va_loss / max(va_n, 1)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch={ep:03d} train={tr_loss:.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Contracted checkpoint format (schema_version=2)
    ckpt = {
        "schema_version": 2,
        "feature_spec": {
            "version": int(FEATURE_SPEC_V1.version),
            "action_encoding": str(FEATURE_SPEC_V1.action_encoding),
            "dim": int(FEATURE_SPEC_V1.dim),
            "window": int(window) if window is not None else None,
        },
        "target_spec": {
            "version": int(TARGET_SPEC_V1.version),
            "mode": str(TARGET_SPEC_V1.mode),
            "dim": int(TARGET_SPEC_V1.dim),
        },
        "model_state": model.state_dict(),
        "in_dim": int(X_train.shape[1]),
        "out_dim": int(Y_train.shape[1]),
        "hidden": int(args.hidden),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "seed": int(args.seed),
    }

    torch.save(ckpt, str(out_path))

    meta = {
        "data": str(data_path),
        "out": str(out_path),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden": int(args.hidden),
        "best_val": float(best_val),
        "schema_version": 2,
        "feature_dim": int(X_train.shape[1]),
        "target_dim": int(Y_train.shape[1]),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK")
    print("ckpt:", str(out_path))
    print("meta:", str(meta_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
