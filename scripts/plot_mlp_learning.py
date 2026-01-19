from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot MLP learning curves from a checkpoint.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/plots"))
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])

    if not train_losses:
        raise SystemExit("Checkpoint missing train_losses.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) Loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_mse")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("MLP training curves")
    plt.legend()
    out1 = args.outdir / "mlp_loss_curves.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    print("OK")
    print("wrote:", str(out1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
