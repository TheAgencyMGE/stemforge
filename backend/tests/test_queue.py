"""Durable queue semantics: claiming, leases, retries, cancellation."""

from __future__ import annotations

from datetime import timedelta

from stemforge import queue
from stemforge.db import session_scope
from stemforge.models import Job, Project, utcnow


def _project(db) -> Project:
    p = Project(name="t", original_filename="t.wav", source_format="wav", options={})
    db.add(p)
    db.flush()
    return p


def test_claim_is_exclusive_and_ordered(env):
    with session_scope() as db:
        p = _project(db)
        low = queue.enqueue(db, p.id, "process", {}, priority=100)
        high = queue.enqueue(db, p.id, "retranscribe", {}, priority=10)
        ids = (low.id, high.id)
    with session_scope() as db:
        first = queue.claim_next(db, "w1")
        second = queue.claim_next(db, "w2")
        third = queue.claim_next(db, "w3")
        assert first.id == ids[1] and second.id == ids[0] and third is None
        assert first.lease_owner == "w1" and first.attempts == 1


def test_retry_with_backoff_then_permanent_failure(env):
    with session_scope() as db:
        p = _project(db)
        job = queue.enqueue(db, p.id, "process", {})
        job.max_attempts = 2
        jid = job.id
    with session_scope() as db:
        j = queue.claim_next(db, "w")
        assert queue.fail(db, j, "boom", retryable=True) is True
        assert db.get(Job, jid).status == "queued"
    with session_scope() as db:
        j = queue.claim_next(db, "w")  # retry_base_seconds=0 in tests, so immediately runnable
        assert j is not None and j.attempts == 2
        assert queue.fail(db, j, "boom again", retryable=True) is False
        assert db.get(Job, jid).status == "failed"


def test_expired_lease_is_reclaimed(env):
    with session_scope() as db:
        p = _project(db)
        queue.enqueue(db, p.id, "process", {})
    with session_scope() as db:
        j = queue.claim_next(db, "crashed-worker")
        j.lease_expires_at = utcnow() - timedelta(seconds=5)
        db.commit()
    with session_scope() as db:
        j2 = queue.claim_next(db, "w2")
        assert j2 is not None and j2.lease_owner == "w2" and j2.attempts == 2


def test_cancel_queued_and_running(env):
    with session_scope() as db:
        p = _project(db)
        a = queue.enqueue(db, p.id, "process", {})
        queue.request_cancel(db, a)
        assert a.status == "cancelled"
        b = queue.enqueue(db, p.id, "process", {})
        bid = b.id
    with session_scope() as db:
        j = queue.claim_next(db, "w")
        assert j.id == bid
        queue.request_cancel(db, j)
        try:
            queue.heartbeat(db, j, "w")
            raise AssertionError("expected JobCancelled")
        except queue.JobCancelled:
            pass
        queue.requeue_for_resume(db, j)
        assert j.status == "queued" and not j.cancel_requested and j.attempts == 0
