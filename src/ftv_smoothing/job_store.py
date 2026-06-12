"""SQLite-backed job state for the FastAPI and Celery web application."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .app_core import TERMINAL_STATUSES, has_completed_files, now_iso, public_result_metadata


class JobStore:
    """Persist Celery job state and result metadata in SQLite."""

    def __init__(self, db_path: str | Path, results_dir: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.results_dir = Path(results_dir).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    celery_task_id TEXT,
                    status TEXT NOT NULL,
                    progress_percent REAL NOT NULL,
                    stage TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    input_fingerprint_json TEXT NOT NULL,
                    result_metadata_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_cache_status ON jobs(cache_key, status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)"
            )

    def create_job(
        self,
        *,
        cache_key: str,
        parameters: dict[str, Any],
        input_fingerprint: dict[str, Any],
        status: str = "queued",
        progress_percent: float = 0.0,
        stage: str = "Queued",
        result_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a job row and return it."""

        job_id = uuid.uuid4().hex
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, cache_key, celery_task_id, status, progress_percent, stage,
                    parameters_json, input_fingerprint_json, result_metadata_json,
                    error, created_at, updated_at
                )
                VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    job_id,
                    cache_key,
                    status,
                    float(progress_percent),
                    stage,
                    json.dumps(parameters, sort_keys=True, ensure_ascii=False),
                    json.dumps(input_fingerprint, sort_keys=True, ensure_ascii=False),
                    json.dumps(result_metadata, sort_keys=True, ensure_ascii=False)
                    if result_metadata
                    else None,
                    timestamp,
                    timestamp,
                ),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def set_celery_task_id(self, job_id: str, celery_task_id: str) -> None:
        """Attach a Celery task id to an existing job row."""

        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET celery_task_id = ?, updated_at = ? WHERE id = ?",
                (celery_task_id, now_iso(), job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Read one job by id."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def find_active_by_cache_key(self, cache_key: str) -> dict[str, Any] | None:
        """Return the latest queued/running job for this cache key."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE cache_key = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent jobs ordered by update time."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_progress(
        self,
        job_id: str,
        *,
        status: str = "running",
        progress_percent: float,
        stage: str,
        error: str | None = None,
    ) -> None:
        """Update mutable progress fields for a job."""

        progress = max(0.0, min(100.0, float(progress_percent)))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, progress_percent = ?, stage = ?, error = ?, updated_at = ?
                WHERE id = ? AND status != 'cancelled'
                """,
                (status, progress, stage, error, now_iso(), job_id),
            )

    def complete_job(
        self,
        job_id: str,
        *,
        result_metadata: dict[str, Any],
        stage: str = "Completed",
    ) -> None:
        """Mark a job completed and store result metadata."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', progress_percent = 100, stage = ?,
                    result_metadata_json = ?, error = NULL, updated_at = ?
                WHERE id = ? AND status != 'cancelled'
                """,
                (
                    stage,
                    json.dumps(result_metadata, sort_keys=True, ensure_ascii=False),
                    now_iso(),
                    job_id,
                ),
            )

    def fail_job(
        self,
        job_id: str,
        error: str,
        *,
        result_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a job failed."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', stage = 'Failed', error = ?,
                    result_metadata_json = COALESCE(?, result_metadata_json),
                    updated_at = ?
                WHERE id = ? AND status != 'cancelled'
                """,
                (
                    error,
                    json.dumps(result_metadata, sort_keys=True, ensure_ascii=False)
                    if result_metadata
                    else None,
                    now_iso(),
                    job_id,
                ),
            )

    def cancel_job(
        self,
        job_id: str,
        *,
        reason: str = "Cancelled by user",
        result_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Mark a queued/running job cancelled and return the current row."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', stage = 'Cancelled', error = ?,
                    result_metadata_json = COALESCE(?, result_metadata_json),
                    updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (
                    reason,
                    json.dumps(result_metadata, sort_keys=True, ensure_ascii=False)
                    if result_metadata
                    else None,
                    now_iso(),
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        """Return True when a job has been cancelled."""

        job = self.get_job(job_id)
        return bool(job and job["status"] == "cancelled")

    def delete_job(self, job_id: str) -> dict[str, Any] | None:
        """Delete a job row and return the deleted row."""

        job = self.get_job(job_id)
        if job is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return job

    def count_jobs_for_cache_key(self, cache_key: str) -> int:
        """Count job rows that still reference a cache key."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def referenced_cache_keys(self) -> set[str]:
        """Return all cache keys referenced by job rows."""

        with self._connect() as connection:
            rows = connection.execute("SELECT DISTINCT cache_key FROM jobs").fetchall()
        return {str(row["cache_key"]) for row in rows}

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Return a browser-facing representation for one job."""

        public = dict(job)
        metadata = job.get("result_metadata")
        public["terminal"] = job["status"] in TERMINAL_STATUSES
        public["can_cancel"] = job["status"] not in TERMINAL_STATUSES
        public["can_retry"] = job["status"] in {"failed", "cancelled"}
        public["can_delete"] = True
        public["error_details"] = metadata.get("traceback") if isinstance(metadata, dict) else None
        public["result"] = None
        if metadata and has_completed_files(self.results_dir, metadata):
            public["result"] = public_result_metadata(
                metadata,
                cache_hit=bool(metadata.get("cache_hit")),
            )
        return public

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        parameters = json.loads(row["parameters_json"])
        input_fingerprint = json.loads(row["input_fingerprint_json"])
        result_metadata = (
            json.loads(row["result_metadata_json"]) if row["result_metadata_json"] else None
        )
        return {
            "id": row["id"],
            "cache_key": row["cache_key"],
            "celery_task_id": row["celery_task_id"],
            "status": row["status"],
            "progress_percent": row["progress_percent"],
            "stage": row["stage"],
            "parameters": parameters,
            "input_fingerprint": input_fingerprint,
            "result_metadata": result_metadata,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
