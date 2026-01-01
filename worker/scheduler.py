from __future__ import annotations

import time
from datetime import datetime, timedelta

from db import SessionLocal
from models import ConnectionStatus, MailboxConnection, Provider
from services.queue import enqueue_job
from sqlmodel import select

POLL_INTERVAL_SECONDS = 300  # 5 minutes
RENEW_INTERVAL_SECONDS = 1800  # 30 minutes


def run_scheduler() -> None:
    last_renew = 0
    while True:
        now = time.time()
        with SessionLocal() as session:
            connections = session.exec(
                select(MailboxConnection).where(MailboxConnection.status == ConnectionStatus.active)
            ).all()
            for conn in connections:
                # if expiration is near, enqueue renew first
                exp = None
                try:
                    meta = json.loads(conn.provider_metadata or "{}")
                    exp_str = meta.get("expiration")
                    if exp_str:
                        exp = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                except Exception:
                    exp = None
                if exp and exp <= datetime.utcnow() + timedelta(minutes=15):
                    enqueue_job(session, "renew_watch", {"connection_id": conn.id})
                else:
                    enqueue_job(session, "poll_connection", {"connection_id": conn.id})
                if now - last_renew > RENEW_INTERVAL_SECONDS:
                    enqueue_job(session, "renew_watch", {"connection_id": conn.id})
        if now - last_renew > RENEW_INTERVAL_SECONDS:
            last_renew = now
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_scheduler()
