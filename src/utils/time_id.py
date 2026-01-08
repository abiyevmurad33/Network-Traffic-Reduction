from __future__ import annotations

from datetime import UTC, datetime


def utc_timestamp_for_id() -> str:
    """
    Return a UTC timestamp safe for folder names:
    YYYY-MM-DDTHH-MM-SSZ (note: ':' replaced with '-')
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def make_session_id(prefix: str = "session") -> str:
    """
    Create a session id per protocol: session_<YYYY-MM-DDTHH-MM-SSZ>
    """
    return f"{prefix}_{utc_timestamp_for_id()}"


def utc_now_iso() -> str:
    """
    Return ISO-8601 UTC timestamp (with trailing Z).
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _self_test() -> None:
    sid = make_session_id()
    assert sid.startswith("session_")
    iso = utc_now_iso()
    assert iso.endswith("Z")
    print("time_id.py self-test: OK")


if __name__ == "__main__":
    _self_test()
