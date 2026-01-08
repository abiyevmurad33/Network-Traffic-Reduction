from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    states_csv: Path
    session_json: Path
    states_csv_tmp: Path
    session_json_tmp: Path


def repo_root_from_file(file_path: str) -> Path:
    """
    Given a file path inside the repo (e.g., this module), return repo root.
    Assumes structure: <root>/src/utils/paths.py
    """
    p = Path(file_path).resolve()
    # .../src/utils/paths.py -> .../src/utils -> .../src -> .../<root>
    return p.parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_replace(src: Path, dst: Path) -> None:
    """
    Atomic replace on same filesystem (Windows-safe).
    """
    os.replace(str(src), str(dst))


def build_session_paths(output_root: Path, session_id: str) -> SessionPaths:
    session_dir = output_root / session_id
    states_csv = session_dir / "states.csv"
    session_json = session_dir / "session.json"
    states_csv_tmp = session_dir / "states.csv.tmp"
    session_json_tmp = session_dir / "session.json.tmp"
    return SessionPaths(
        session_dir=session_dir,
        states_csv=states_csv,
        session_json=session_json,
        states_csv_tmp=states_csv_tmp,
        session_json_tmp=session_json_tmp,
    )


def _self_test() -> None:
    root = Path(".").resolve()
    sp = build_session_paths(root / "data" / "raw_logs", "session_test")
    assert sp.states_csv.name == "states.csv"
    assert sp.session_json_tmp.name.endswith(".tmp")
    print("paths.py self-test: OK")


if __name__ == "__main__":
    _self_test()
