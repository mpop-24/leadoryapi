from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select

from models import Job, JobStatus


def enqueue_job(session: Session, job_type: str, payload: dict[str, Any], scheduled_at: Optional[datetime] = None) -> Job:
    job = Job(job_type=job_type, payload=json.dumps(payload), scheduled_at=scheduled_at)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def fetch_next_job(session: Session) -> Optional[Job]:
    stmt = select(Job).where(Job.status == JobStatus.pending).order_by(Job.created_at.asc())
    job = session.exec(stmt).first()
    if job:
        job.status = JobStatus.in_progress
        job.attempts += 1
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def complete_job(session: Session, job: Job) -> None:
    job.status = JobStatus.completed
    session.add(job)
    session.commit()


def fail_job(session: Session, job: Job, error: str) -> None:
    job.status = JobStatus.failed
    job.last_error = error[:500]
    session.add(job)
    session.commit()


# Simple blocking worker loop

def worker_loop(processor, poll_interval: float = 1.0) -> None:
    while True:
        with processor.session_factory() as session:
            job = fetch_next_job(session)
            if not job:
                time.sleep(poll_interval)
                continue
            try:
                processor.process(job, session)
                complete_job(session, job)
            except Exception as exc:  # pragma: no cover
                fail_job(session, job, str(exc))
                time.sleep(poll_interval)
