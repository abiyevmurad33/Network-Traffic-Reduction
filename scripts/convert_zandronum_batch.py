from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch convert Zandronum client logs to session folders."
    )
    ap.add_argument("--logs-dir", type=Path, default=Path("zandronum/logs"))
    ap.add_argument("--out-root", type=Path, default=Path("data/raw_logs"))
    ap.add_argument("--pattern", type=str, default="client_zan_*.log")
    ap.add_argument("--tps", type=float, default=35.0)
    ap.add_argument("--write-session-json", action="store_true")
    args = ap.parse_args()

    logs = sorted(args.logs_dir.glob(args.pattern))
    if not logs:
        raise SystemExit(f"No logs matched {args.logs_dir / args.pattern}")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    converted = 0
    for log in logs:
        # Stable session id based on log filename + UTC timestamp now
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        session_dir = out_root / f"session_zan_{log.stem}_{ts}"

        cmd = [
            "python",
            "-m",
            "scripts.convert_zandronum_log",
            "--log",
            str(log),
            "--out",
            str(session_dir),
            "--tps",
            str(args.tps),
        ]
        if args.write_session_json:
            cmd.append("--write-session-json")

        print("RUN:", " ".join(cmd))
        subprocess.check_call(cmd)
        converted += 1

    print("OK")
    print("logs_found:", len(logs))
    print("converted:", converted)
    print("out_root:", str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
