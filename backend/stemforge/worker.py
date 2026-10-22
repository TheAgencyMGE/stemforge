"""Background worker: claims jobs from the DB queue and runs the ML pipeline outside the web server."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
import traceback

from sqlalchemy.orm import Session

from . import queue
from .config import get_settings
from .db import get_sessionmaker
from .logging_setup import configure_logging, job_id_var
from .models import Job, Project
from .storage import get_storage

log = logging.getLogger(__name__)
_stop = threading.Event()


def _heartbeat_loop(job_id: str, worker_id: str, done: threading.Event) -> None:
    """Keep the lease alive during long, uninterruptible steps (e.g. one big inference call)."""
    settings = get_settings()
    session = get_sessionmaker()()
    try:
        while not done.wait(settings.job_heartbeat_seconds):
            job = session.get(Job, job_id)
            if job is None or job.status != "running" or job.lease_owner != worker_id:
                return
            from datetime import timedelta

            from .models import utcnow

            job.lease_expires_at = utcnow() + timedelta(seconds=settings.job_lease_seconds)
            session.commit()
            session.expire_all()
    except Exception:  # noqa: BLE001
        log.exception("heartbeat thread error")
    finally:
        session.close()


def process_one(worker_id: str) -> bool:
    """Claim and execute a single job. Returns False when the queue was empty."""
    from .pipeline.runner import PermanentError, run_job

    session: Session = get_sessionmaker()()
    try:
        job = queue.claim_next(session, worker_id)
        if job is None:
            return False
        token = job_id_var.set(job.id)
        done = threading.Event()
        hb = threading.Thread(target=_heartbeat_loop, args=(job.id, worker_id, done), daemon=True)
        hb.start()
        t0 = time.monotonic()
        log.info("job started", extra={"kind": job.kind, "attempt": job.attempts, "project_id": job.project_id})
        try:
            run_job(session, job, get_storage(), worker_id)
            queue.complete(session, job)
            log.info("job succeeded", extra={"seconds": round(time.monotonic() - t0, 2)})
        except queue.JobCancelled as exc:
            session.rollback()
            session.refresh(job)
            if job.lease_owner == worker_id or job.cancel_requested:
                queue.mark_cancelled(session, job)
                _set_project_status(session, job, "cancelled", "Processing was cancelled.")
            log.info("job cancelled", extra={"reason": str(exc)})
        except PermanentError as exc:
            session.rollback()
            session.refresh(job)
            queue.fail(session, job, str(exc), retryable=False)
            _set_project_status(session, job, "failed", str(exc))
            log.warning("job failed permanently", extra={"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - unknown errors are retried with backoff
            session.rollback()
            session.refresh(job)
            detail = f"{type(exc).__name__}: {exc}"
            retried = queue.fail(session, job, detail, retryable=True)
            if not retried:
                _set_project_status(session, job, "failed", f"Processing failed: {detail}")
            log.error("job error", extra={"retrying": retried, "trace": traceback.format_exc()[-4000:]})
        finally:
            done.set()
            job_id_var.reset(token)
        return True
    finally:
        session.close()


def _set_project_status(session: Session, job: Job, status: str, error: str | None) -> None:
    project = session.get(Project, job.project_id)
    if project is None:
        return
    if job.kind == "retranscribe":
        # A failed re-transcription leaves the existing (ready) project intact.
        project.error = error if status == "failed" else None
    else:
        project.status = status
        project.error = error
    session.commit()


def run_loop(worker_id: str, poll_seconds: float = 1.0) -> None:
    configure_logging()
    log.info("worker started", extra={"worker_id": worker_id})
    _warmup()
    last_cleanup = 0.0
    while not _stop.is_set():
        try:
            if time.monotonic() - last_cleanup > 600:
                last_cleanup = time.monotonic()
                _cleanup()
            if not process_one(worker_id):
                _stop.wait(poll_seconds)
        except Exception:  # noqa: BLE001 - e.g. transient DB errors; keep the worker alive
            log.exception("worker loop error")
            _stop.wait(5)
    log.info("worker stopped")


def _cleanup() -> None:
    from .db import session_scope
    from .services import cleanup_expired

    try:
        with session_scope() as db:
            n = cleanup_expired(db)
        if n:
            log.info("retention cleanup", extra={"deleted": n})
    except Exception:  # noqa: BLE001
        log.exception("retention cleanup failed")


def _warmup() -> None:
    try:
        from .pipeline.separation import resolve_device

        log.info("inference device", extra={"device": resolve_device()})
    except Exception as exc:  # noqa: BLE001
        log.warning("device probe failed", extra={"error": str(exc)})
    # librosa's numba kernels JIT-compile on first use (~30 s); pay that once at startup, not per job.
    try:
        import numpy as np

        from .pipeline import analysis, transcription

        t0 = time.monotonic()
        noise = (np.random.default_rng(0).standard_normal((2, 44100 * 25)) * 0.05).astype(np.float32)
        analysis.analyze(noise, 44100)
        transcription._bp_model()
        log.info("warmup done", extra={"seconds": round(time.monotonic() - t0, 1)})
    except Exception as exc:  # noqa: BLE001
        log.warning("warmup failed", extra={"error": str(exc)})


def _child(index: int) -> None:
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    run_loop(f"{socket.gethostname()}-{os.getpid()}-{index}")


def main(concurrency: int | None = None) -> None:
    configure_logging()
    from .migrate import upgrade

    upgrade()
    n = concurrency or get_settings().worker_concurrency
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    if n <= 1:
        run_loop(f"{socket.gethostname()}-{os.getpid()}")
        return
    procs = [mp.Process(target=_child, args=(i,), daemon=False) for i in range(n)]
    for p in procs:
        p.start()
    try:
        while not _stop.is_set() and any(p.is_alive() for p in procs):
            _stop.wait(1)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(10)
