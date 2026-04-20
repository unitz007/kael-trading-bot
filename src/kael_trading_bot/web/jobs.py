"""Training job state management.

Tracks in-flight training jobs in memory so the API can report status
without hitting the filesystem or requiring a database.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TrainingStatus(str, Enum):
    """Possible states for a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Represents a single training job."""

    job_id: str
    pair: str
    status: TrainingStatus = TrainingStatus.PENDING
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pair": self.pair,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class TrainingJobStore:
    """Thread-safe in-memory store for training jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create(self, pair: str) -> TrainingJob:
        """Create a new training job for *pair*."""
        job = TrainingJob(job_id=str(uuid.uuid4()), pair=pair)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> TrainingJob | None:
        """Return a job by ID, or ``None`` if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> TrainingJob | None:
        """Update fields on an existing job.  Returns updated job or ``None``."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return job

    def list_all(self) -> list[TrainingJob]:
        """Return all jobs, newest first."""
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )

    def get_running_for_pair(self, pair: str) -> TrainingJob | None:
        """Return an active (pending/running) job for *pair*, if any."""
        with self._lock:
            for job in self._jobs.values():
                if job.pair == pair and job.status in (
                    TrainingStatus.PENDING,
                    TrainingStatus.RUNNING,
                ):
                    return job
        return None

    def clear(self) -> None:
        """Remove all jobs (useful for testing)."""
        with self._lock:
            self._jobs.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store = TrainingJobStore()


def get_job_store() -> TrainingJobStore:
    """Return the global :class:`TrainingJobStore` singleton."""
    return _store
