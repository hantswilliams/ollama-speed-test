"""SQLite logging for the Ollama proxy."""

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "usage.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    streamed INTEGER NOT NULL,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    prompt_eval_duration_ns INTEGER,
    eval_duration_ns INTEGER,
    load_duration_ns INTEGER,
    total_duration_ns INTEGER,
    output_tps REAL,
    prompt_tps REAL,
    wall_time_sec REAL,
    client_ip TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
"""

_write_lock = Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def insert_request(record: dict[str, Any]) -> None:
    cols = (
        "timestamp", "model", "endpoint", "streamed",
        "prompt_tokens", "output_tokens",
        "prompt_eval_duration_ns", "eval_duration_ns",
        "load_duration_ns", "total_duration_ns",
        "output_tps", "prompt_tps", "wall_time_sec", "client_ip",
    )
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO requests ({', '.join(cols)}) VALUES ({placeholders})"
    values = tuple(record.get(c) for c in cols)
    with _write_lock, _connect() as conn:
        conn.execute(sql, values)
