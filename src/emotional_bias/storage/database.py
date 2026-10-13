"""
Storage module for persisting benchmark results.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from ..analysis.scoring import ResponseScore
from ..models.base import ModelConfig, ModelResponse
from ..payloads.base import Payload

logger = structlog.get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: Path = Path("results.db")
    wal_mode: bool = True
    foreign_keys: bool = True


class BenchmarkDatabase:
    """SQLite database for storing benchmark results."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS payloads (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    payload_type TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    payload_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    messages TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    usage TEXT,
                    latency_ms REAL NOT NULL,
                    error TEXT,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id),
                    FOREIGN KEY (model_id) REFERENCES models(id),
                    FOREIGN KEY (payload_id) REFERENCES payloads(id)
                );

                CREATE TABLE IF NOT EXISTS scores (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    payload_emotion TEXT,
                    overall_score REAL NOT NULL,
                    dimension_scores TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id),
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );

                CREATE INDEX IF NOT EXISTS idx_responses_run_model ON responses(run_id, model_id);
                CREATE INDEX IF NOT EXISTS idx_responses_payload ON responses(payload_id);
                CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
            """)

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.config.path)
        conn.row_factory = sqlite3.Row
        if self.config.wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        if self.config.foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_run(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new benchmark run."""
        run_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO runs (id, name, description, metadata) VALUES (?, ?, ?, ?)",
                (run_id, name, description, json.dumps(metadata or {})),
            )
        logger.info("run_created", run_id=run_id, name=name)
        return run_id

    def add_model(self, run_id: str, config: ModelConfig) -> str:
        """Add a model configuration to a run."""
        model_id = config.name
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO models
                   (id, run_id, name, provider, model_id, config)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    model_id,
                    run_id,
                    config.name,
                    config.provider.value,
                    config.model_id,
                    json.dumps(config.__dict__, default=str),
                ),
            )
        return model_id

    def add_payload(self, run_id: str, payload: Payload) -> str:
        """Add a payload to a run."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO payloads
                   (id, run_id, emotion, payload_type, intensity, text, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload.id,
                    run_id,
                    payload.config.emotion.value,
                    payload.config.payload_type.value,
                    payload.config.intensity,
                    payload.text,
                    json.dumps(payload.metadata),
                ),
            )
        return payload.id

    def add_response(self, run_id: str, response: ModelResponse) -> str:
        """Add a model response."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO responses
                   (id, run_id, model_id, payload_id, request_id, timestamp,
                    messages, response_text, usage, latency_ms, error, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    response.request_id,
                    run_id,
                    response.model_config.name,
                    response.metadata.get("payload_id", ""),
                    response.request_id,
                    response.timestamp.isoformat(),
                    json.dumps([m.to_dict() for m in response.messages]),
                    response.response_text,
                    json.dumps(response.usage) if response.usage else None,
                    response.latency_ms,
                    response.error,
                    json.dumps(response.metadata),
                ),
            )
        return response.request_id

    def add_score(self, run_id: str, response_id: str, score: ResponseScore) -> str:
        """Add a scoring result."""
        score_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO scores
                   (id, run_id, response_id, payload_emotion, overall_score,
                    dimension_scores, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    score_id,
                    run_id,
                    response_id,
                    score.payload_emotion.value if score.payload_emotion else None,
                    score.overall_score,
                    json.dumps(
                        [
                            {
                                "dimension": ds.dimension.value,
                                "score": ds.score,
                                "confidence": ds.confidence,
                                "evidence": ds.evidence,
                                "metadata": ds.metadata,
                            }
                            for ds in score.dimension_scores
                        ],
                    ),
                    json.dumps(score.metadata),
                ),
            )
        return score_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get run metadata."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def get_models(self, run_id: str) -> list[dict[str, Any]]:
        """Get models for a run."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM models WHERE run_id = ?", (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_payloads(self, run_id: str) -> list[dict[str, Any]]:
        """Get payloads for a run."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM payloads WHERE run_id = ?", (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_responses(
        self,
        run_id: str,
        model_id: str | None = None,
        payload_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get responses for a run."""
        query = "SELECT * FROM responses WHERE run_id = ?"
        params = [run_id]

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if payload_id:
            query += " AND payload_id = ?"
            params.append(payload_id)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_scores(
        self,
        run_id: str,
        emotion: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get scores for a run."""
        query = "SELECT * FROM scores WHERE run_id = ?"
        params = [run_id]

        if emotion:
            query += " AND payload_emotion = ?"
            params.append(emotion)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def export_run_jsonl(self, run_id: str, output_path: Path) -> int:
        """Export run data as JSONL."""
        responses = self.get_responses(run_id)
        scores = {s["response_id"]: s for s in self.get_scores(run_id)}
        payloads = {p["id"]: p for p in self.get_payloads(run_id)}

        count = 0
        with output_path.open("w") as f:
            for resp in responses:
                payload = payloads.get(resp["payload_id"], {})
                score = scores.get(resp["id"], {})

                record = {
                    "run_id": run_id,
                    "model": resp["model_id"],
                    "payload": payload,
                    "response": resp,
                    "score": score,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info("exported_jsonl", run_id=run_id, path=str(output_path), records=count)
        return count


class JSONLStorage:
    """Simple JSONL file storage for streaming results."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self) -> JSONLStorage:
        self._file = self.path.open("a")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._file:
            self._file.close()

    def write(self, record: dict[str, Any]) -> None:
        """Write a record."""
        if self._file:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all records."""
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f]


__all__ = [
    "DatabaseConfig",
    "BenchmarkDatabase",
    "JSONLStorage",
]
