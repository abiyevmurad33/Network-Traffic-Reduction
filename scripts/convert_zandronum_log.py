from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Your confirmed mapping (source-of-truth for the project)
BTN_ATTACK = 1
BTN_USE = 2
BTN_JUMP = 4
BTN_SHIFT = 256
BTN_D = 1024
BTN_A = 2048
BTN_S = 4096
BTN_W = 8192


# Example log line (what we parse):
# [14:03:53] ZTRACE tic=749 x=444.209 y=-213.124 z=-64 yaw_deg=28.6523 pitch_deg=6.15234 buttons=2048
_ZTRACE_RE = re.compile(
    r"ZTRACE\s+tic=(?P<tic>-?\d+)\s+"
    r"x=(?P<x>-?\d+(?:\.\d+)?)\s+"
    r"y=(?P<y>-?\d+(?:\.\d+)?)\s+"
    r"z=(?P<z>-?\d+(?:\.\d+)?)\s+"
    r"yaw_deg=(?P<yaw>-?\d+(?:\.\d+)?)\s+"
    r"pitch_deg=(?P<pitch>-?\d+(?:\.\d+)?)\s+"
    r"buttons=(?P<buttons>-?\d+)"
)


@dataclass(frozen=True)
class SessionMeta:
    source: str
    session_id: str
    map_name: str | None
    tics_per_second: float
    button_map: dict[str, int]


def _guess_session_id_from_path(p: Path) -> str:
    # Use filename stem up to first "__" if present (your logs often have suffixes)
    stem = p.name
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    # strip extension(s)
    return Path(stem).stem


def _extract_map_name(text: str) -> str | None:
    # Example: *** MAP01: Hydroelectric Plant ***
    m = re.search(r"\*\*\*\s*(?P<map>MAP\d+)[^*]*\*\*\*", text)
    if not m:
        return None
    return m.group("map")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert Zandronum ZTRACE logfile to states.csv session folder."
    )
    ap.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to Zandronum client logfile containing ZTRACE lines.",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="Output session directory (will be created)."
    )
    ap.add_argument(
        "--tps", type=float, default=35.0, help="Tics per second (Doom engine default is 35)."
    )
    ap.add_argument(
        "--write-session-json", action="store_true", help="Also write session.json with metadata."
    )
    args = ap.parse_args()

    log_path: Path = args.log
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="replace")

    rows: list[dict[str, str | int | float]] = []
    last_tic: int | None = None

    for line in text.splitlines():
        m = _ZTRACE_RE.search(line)
        if not m:
            continue

        tic = int(m.group("tic"))
        x = float(m.group("x"))
        y = float(m.group("y"))
        z = float(m.group("z"))
        yaw = float(m.group("yaw"))
        pitch = float(m.group("pitch"))
        buttons = int(m.group("buttons"))

        # Keep only strictly increasing tics (guards duplicates / repeated prints)
        if last_tic is not None and tic <= last_tic:
            continue
        last_tic = tic

        rows.append(
            {
                "tic": tic,
                "pos_x": x,
                "pos_y": y,
                "pos_z": z,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "action_mask": buttons,
            }
        )

    if len(rows) < 2:
        raise SystemExit(f"Not enough ZTRACE rows parsed from {log_path} (got {len(rows)})")

    # Write states.csv
    states_csv = out_dir / "states.csv"
    header = "tic,pos_x,pos_y,pos_z,yaw_deg,pitch_deg,action_mask\n"
    with states_csv.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for r in rows:
            f.write(
                f'{r["tic"]},{r["pos_x"]},{r["pos_y"]},{r["pos_z"]},'
                f'{r["yaw_deg"]},{r["pitch_deg"]},{r["action_mask"]}\n'
            )

    # Optional session.json
    if args.write_session_json:
        meta = SessionMeta(
            source="zandronum",
            session_id=_guess_session_id_from_path(log_path),
            map_name=_extract_map_name(text),
            tics_per_second=float(args.tps),
            button_map={
                "ATTACK": BTN_ATTACK,
                "USE": BTN_USE,
                "JUMP": BTN_JUMP,
                "SHIFT": BTN_SHIFT,
                "D": BTN_D,
                "A": BTN_A,
                "S": BTN_S,
                "W": BTN_W,
            },
        )
        (out_dir / "session.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    print("OK")
    print("in:", str(log_path))
    print("out:", str(out_dir))
    print("rows:", len(rows))
    print("first_tic:", rows[0]["tic"], "last_tic:", rows[-1]["tic"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
