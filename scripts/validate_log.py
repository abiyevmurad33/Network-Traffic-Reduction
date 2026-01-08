from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from src.logger.action_space import ACTION_LABELS, canonical_action_space

REQUIRED_CSV_HEADER = [
    "session_id",
    "tic",
    "player_id",
    "pos_x",
    "pos_y",
    "pos_z",
    "yaw_deg",
    "pitch_deg",
    "action_mask",
]


REQUIRED_SESSION_KEYS = [
    "schema_version",
    "session_id",
    "created_utc",
    "engine",
    "scenario_name",
    "scenario_config_path",
    "wad",
    "seed",
    "tics_per_second",
    "duration_seconds",
    "protocol_tier",
    "warmup_seconds",
    "players",
    "action_space",
    "ended_early",
    "tics_recorded",
    "end_reason",
    "first_tic",
    "last_tic",
]


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def _check_angle(name: str, v: float) -> None:
    if not _is_finite(v):
        _fail(f"{name} must be finite, got {v}")
    # canonical range (-180, 180]
    if not (-180.0 < v <= 180.0):
        _fail(f"{name} out of range (-180,180], got {v}")


def _check_float(name: str, v: float) -> None:
    if not _is_finite(v):
        _fail(f"{name} must be finite, got {v}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        _fail(f"Failed to read JSON {path}: {type(e).__name__}: {e}")
    raise AssertionError("unreachable")


def _validate_session_json(meta: dict[str, Any]) -> None:
    # Required keys
    missing = [k for k in REQUIRED_SESSION_KEYS if k not in meta]
    if missing:
        _fail(f"session.json missing keys: {missing}")

    if meta["schema_version"] != "1.0":
        _fail(f"schema_version must be '1.0', got {meta['schema_version']!r}")

    if meta["engine"] != "vizdoom":
        _fail(f"engine must be 'vizdoom', got {meta['engine']!r}")

    if meta["protocol_tier"] not in ("A", "B"):
        _fail(f"protocol_tier must be 'A' or 'B', got {meta['protocol_tier']!r}")

    # Action space
    action_space = meta["action_space"]
    if not isinstance(action_space, dict):
        _fail("action_space must be an object/dict")

    if action_space.get("type") != "multi_binary_bitmask":
        _fail(f"action_space.type must be 'multi_binary_bitmask', got {action_space.get('type')!r}")

    labels = action_space.get("labels")
    if labels != ACTION_LABELS:
        _fail(f"action_space.labels mismatch. Expected {ACTION_LABELS}, got {labels}")

    # Players basic sanity
    players = meta["players"]
    if not isinstance(players, list) or len(players) != 1:
        _fail("players must be a list with exactly 1 entry for v1")

    p0 = players[0]
    if p0.get("player_id") != "p1":
        _fail(f"players[0].player_id must be 'p1', got {p0.get('player_id')!r}")


def _validate_states_csv(path: Path, expected_session_id: str) -> tuple[int, int, int]:
    """
    Returns: (rows, first_tic, last_tic)
    """
    sp = canonical_action_space()

    rows = 0
    first_tic: int | None = None
    last_tic: int | None = None
    last_seen_tic: int | None = None

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header is None:
                _fail("states.csv is empty (no header)")
            if header != REQUIRED_CSV_HEADER:
                _fail(
                    f"states.csv header mismatch.\nExpected: {REQUIRED_CSV_HEADER}\nGot:      {header}"
                )

            for line_no, row in enumerate(r, start=2):
                if len(row) != len(REQUIRED_CSV_HEADER):
                    _fail(
                        f"line {line_no}: expected {len(REQUIRED_CSV_HEADER)} cols, got {len(row)}"
                    )

                sid = row[0]
                if sid != expected_session_id:
                    _fail(
                        f"line {line_no}: session_id mismatch: {sid!r} != {expected_session_id!r}"
                    )

                try:
                    tic = int(row[1])
                except Exception:
                    _fail(f"line {line_no}: tic not int: {row[1]!r}")

                if last_seen_tic is not None and tic <= last_seen_tic:
                    _fail(f"line {line_no}: tic not strictly increasing: {tic} <= {last_seen_tic}")
                last_seen_tic = tic

                player_id = row[2]
                if player_id != "p1":
                    _fail(f"line {line_no}: player_id must be 'p1', got {player_id!r}")

                try:
                    pos_x = float(row[3])
                    pos_y = float(row[4])
                    pos_z = float(row[5])
                    yaw = float(row[6])
                    pitch = float(row[7])
                    mask = int(row[8])
                except Exception as e:  # noqa: BLE001
                    _fail(f"line {line_no}: parse error: {type(e).__name__}: {e}")

                _check_float("pos_x", pos_x)
                _check_float("pos_y", pos_y)
                _check_float("pos_z", pos_z)
                _check_angle("yaw_deg", yaw)
                _check_angle("pitch_deg", pitch)

                sp.validate_mask(mask)

                if first_tic is None:
                    first_tic = tic
                last_tic = tic
                rows += 1

    except Exception:
        raise

    if rows == 0:
        _fail("states.csv contains no data rows")

    assert first_tic is not None
    assert last_tic is not None
    return rows, first_tic, last_tic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate a logged session directory.")
    p.add_argument(
        "session_dir",
        type=str,
        help="Path to data/raw_logs/<session_id>/ directory containing states.csv and session.json",
    )
    args = p.parse_args(argv)

    session_dir = Path(args.session_dir).resolve()
    states_csv = session_dir / "states.csv"
    session_json = session_dir / "session.json"

    if not session_dir.exists():
        print(f"FAIL: session_dir does not exist: {session_dir}", file=sys.stderr)
        return 2
    if not states_csv.exists():
        print(f"FAIL: missing states.csv at {states_csv}", file=sys.stderr)
        return 2
    if not session_json.exists():
        print(f"FAIL: missing session.json at {session_json}", file=sys.stderr)
        return 2

    try:
        meta = _load_json(session_json)
        _validate_session_json(meta)
        expected_sid = meta["session_id"]

        rows, first_tic, last_tic = _validate_states_csv(states_csv, expected_sid)

        # Cross-check counts
        if int(meta["tics_recorded"]) != int(rows):
            _fail(
                f"tics_recorded mismatch: session.json={meta['tics_recorded']} vs csv_rows={rows}"
            )

        # Cross-check first/last
        if meta.get("first_tic") is not None and int(meta["first_tic"]) != int(first_tic):
            _fail(f"first_tic mismatch: session.json={meta['first_tic']} vs csv={first_tic}")
        if meta.get("last_tic") is not None and int(meta["last_tic"]) != int(last_tic):
            _fail(f"last_tic mismatch: session.json={meta['last_tic']} vs csv={last_tic}")

    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print("PASS")
    print("session_dir:", str(session_dir))
    print("rows:", rows)
    print("first_tic:", first_tic)
    print("last_tic:", last_tic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
