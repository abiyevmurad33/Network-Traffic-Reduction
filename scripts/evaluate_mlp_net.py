from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_dr_net import load_states  # reuse your loader
from src.eval.net_sim import DRConfig, NetConfig, evaluate_predictor_under_network_factory
from src.predictors.mlp_predictor import MLPPredictor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=str)
    ap.add_argument("--ckpt", type=str, default="models/mlp_v1_w8_h1.pt")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    session_dir = Path(args.session_dir).resolve()
    states, action_masks = load_states(session_dir)

    # Use the same defaults you already use in evaluate_dr_net.py
    dr_cfg = DRConfig(update_interval_tics=7, pos_eps=5.0, yaw_eps_deg=5.0, pitch_eps_deg=5.0)
    net_cfg = NetConfig(
        tics_per_second=35.0,
        latency_ms=50.0,
        jitter_ms=5.0,
        loss_rate=0.01,
        bandwidth_kbps=0.0,
        header_bytes=28,
        payload_bytes_state=32,
        rng_seed=int(args.seed),
    )

    ckpt_path = Path(args.ckpt).resolve()

    summary = evaluate_predictor_under_network_factory(
        states=states,
        action_masks=action_masks,
        predictor_factory=lambda: MLPPredictor(ckpt_path=ckpt_path),
        dr_cfg=dr_cfg,
        net_cfg=net_cfg,
    )

    out = session_dir / "evaluation_mlp_net.json"
    out.write_text(json.dumps(summary, default=lambda o: o.__dict__, indent=2), encoding="utf-8")

    print("OK")
    print("out:", str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
