"""原子会话审计、按时点检索的预测反馈；不会修改提示词或风控代码。"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def stamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("记忆时间必须带时区")
    from datetime import UTC

    return value.astimezone(UTC).isoformat()


class AuditStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                  run_id TEXT, session_id TEXT, available_at TEXT, payload TEXT,
                  PRIMARY KEY(run_id, session_id));
                CREATE TABLE IF NOT EXISTS memories (
                  run_id TEXT, memory_id TEXT, available_at TEXT, payload TEXT,
                  PRIMARY KEY(run_id, memory_id));
                CREATE TABLE IF NOT EXISTS decisions (
                  run_id TEXT, decision_id INTEGER, as_of TEXT, payload TEXT,
                  PRIMARY KEY(run_id, decision_id));
                CREATE TABLE IF NOT EXISTS forecasts (
                  run_id TEXT, forecast_id TEXT, due_at TEXT, payload TEXT,
                  PRIMARY KEY(run_id, forecast_id));
            """)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record_session(
        self,
        run_id: str,
        session_id: str,
        available_at: datetime,
        payload: dict[str, Any],
        memories: list[dict[str, Any]] | None = None,
    ) -> None:
        when = stamp(available_at)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", [run_id, session_id, when, encoded])
            for memory in memories or []:
                db.execute(
                    "INSERT INTO memories VALUES (?,?,?,?)",
                    [run_id, memory["id"], when, json.dumps(memory, allow_nan=False)],
                )

    def memories_as_of(self, run_id: str, cutoff: datetime) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM memories WHERE run_id=? AND available_at<=? "
                "ORDER BY available_at,memory_id",
                [run_id, stamp(cutoff)],
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recent_memories_as_of(
        self, run_id: str, cutoff: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """memories_as_of(...)[-limit:]，只解码需要的行。"""
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM memories WHERE run_id=? AND available_at<=? "
                "ORDER BY available_at DESC,memory_id DESC LIMIT ?",
                [run_id, stamp(cutoff), limit],
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def count_memories_as_of(self, run_id: str, cutoff: datetime) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM memories WHERE run_id=? AND available_at<=?",
                [run_id, stamp(cutoff)],
            ).fetchone()
        return int(row[0])

    def save_forecast(
        self, run_id: str, forecast_id: str, due_at: datetime, payload: dict[str, Any]
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO forecasts VALUES (?,?,?,?)",
                [run_id, forecast_id, stamp(due_at), json.dumps(payload, allow_nan=False)],
            )

    def record_decision(
        self, run_id: str, decision_id: int, as_of: datetime, payload: dict[str, Any]
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO decisions VALUES (?,?,?,?)",
                [
                    run_id,
                    decision_id,
                    stamp(as_of),
                    json.dumps(payload, ensure_ascii=False, allow_nan=False),
                ],
            )
