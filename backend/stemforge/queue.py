"""Durable database-backed job queue with leases, heartbeats, retries and cancellation.

Claiming is atomic: on Postgres via ``FOR UPDATE SKIP LOCKED``; on SQLite via a conditional
UPDATE (SQLite serialises writers, so the row-count check is sufficient).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from .config import get_settings
from .db import is_postgres
from .models import Job, utcnow

log = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}


class JobCancelled(Exception):
    pass


class QueueFull(Exception):
    pass


def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)  # SQLite drops tzinfo
    return dt


def enqueue(db: Session, project_id: str, kind: str, params: dict, priority: int = 100) -> Job:
    settings = get_settings()
    active = db.scalar(
        select(Job.id).where(Job.status.in_(["queued", "running"])).limit(settings.max_queued_jobs).offset(
            settings.max_queued_jobs - 1
        )
    )
    if active is not None:
        raise QueueFull(f"queue limit of {settings.max_queued_jobs} active jobs reached")
    job = Job(
        project_id=project_id, kind=kind, params=params, priority=priority,
        max_attempts=settings.job_max_attempts, status="queued", run_after=utcnow(),
    )
    db.add(job)
    db.flush()
    return job


def claim_next(db: Session, worker_id: str) -> Job | None:
    """Atomically claim the next runnable job (queued, or running with an expired lease)."""
    settings = get_settings()
    now = utcnow()
    lease_until = now + timedelta(seconds=settings.job_lease_seconds)
    runnable = or_(
        and_(Job.status == "queued", Job.run_after <= now),
        and_(Job.status == "running", Job.lease_expires_at < now),  # crashed worker
    )
    for _ in range(5):
        q = select(Job.id).where(runnable).order_by(Job.priority, Job.created_at).limit(1)
        if is_postgres():
            q = q.with_for_update(skip_locked=True)
        job_id = db.scalar(q)
        if job_id is None:
            db.rollback()
            return None
        res = db.execute(
            update(Job)
            .where(Job.id == job_id, runnable)
            .values(
                status="running", lease_owner=worker_id, lease_expires_at=lease_until,
                attempts=Job.attempts + 1, started_at=now, error=None,
            )
            .execution_options(synchronize_session=False)
        )
        db.commit()
        if res.rowcount == 1:  # type: ignore[attr-defined]
            job = db.get(Job, job_id)
            assert job is not None
            db.refresh(job)
            if job.attempts > job.max_attempts:
                job.status = "failed"
                job.error = job.error or "exceeded max attempts (worker crashed repeatedly)"
                job.finished_at = utcnow()
                db.commit()
                continue
            return job
    return None


def heartbeat(db: Session, job: Job, worker_id: str) -> None:
    """Extend the lease; raise JobCancelled if cancellation was requested or the lease was lost."""
    settings = get_settings()
    db.refresh(job)
    if job.lease_owner != worker_id or job.status != "running":
        raise JobCancelled("lease lost")
    if job.cancel_requested:
        raise JobCancelled("cancelled by user")
    job.lease_expires_at = utcnow() + timedelta(seconds=settings.job_lease_seconds)
    db.commit()


def complete(db: Session, job: Job) -> None:
    job.status = "succeeded"
    job.progress = 1.0
    job.eta_seconds = None
    job.finished_at = utcnow()
    job.lease_owner = None
    job.lease_expires_at = None
    db.commit()


def mark_cancelled(db: Session, job: Job) -> None:
    job.status = "cancelled"
    job.eta_seconds = None
    job.finished_at = utcnow()
    job.lease_owner = None
    job.lease_expires_at = None
    db.commit()


def fail(db: Session, job: Job, error: str, retryable: bool) -> bool:
    """Record a failure. Returns True when the job was rescheduled for another attempt."""
    settings = get_settings()
    job.error = error[:4000]
    job.lease_owner = None
    job.lease_expires_at = None
    job.eta_seconds = None
    if retryable and job.attempts < job.max_attempts and not job.cancel_requested:
        delay = min(settings.job_retry_base_seconds * (2 ** (job.attempts - 1)), 600)
        job.status = "queued"
        job.run_after = utcnow() + timedelta(seconds=delay)
        job.message = f"attempt {job.attempts} failed; retrying in {int(delay)}s"
        db.commit()
        return True
    job.status = "failed"
    job.finished_at = utcnow()
    db.commit()
    return False


def request_cancel(db: Session, job: Job) -> None:
    if job.status in TERMINAL:
        return
    job.cancel_requested = True
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = utcnow()
    db.commit()


def requeue_for_resume(db: Session, job: Job) -> None:
    """Manual retry of a failed/cancelled job. Finished stages are skipped via artifact hashes."""
    job.status = "queued"
    job.cancel_requested = False
    job.attempts = 0
    job.error = None
    job.message = "resuming"
    job.run_after = utcnow()
    job.finished_at = None
    db.commit()


def lease_expired(job: Job) -> bool:
    exp = _aware(job.lease_expires_at)
    return exp is not None and exp < utcnow()
