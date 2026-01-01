from __future__ import annotations

import json
import time

from db import SessionLocal
from models import Job, ConnectionStatus, Provider
from services.pipeline import process_inbound
from services.queue import worker_loop, enqueue_job
from sqlmodel import Session, select
from models import Client, MailboxConnection
from providers.gmail import GmailProvider
from providers.microsoft import MicrosoftProvider
from services.crypto import decrypt


class Processor:
    def session_factory(self):
        return SessionLocal()

    def process(self, job: Job, session: Session) -> None:
        payload = json.loads(job.payload)
        if job.job_type == "process_inbound":
            connection = session.get(MailboxConnection, payload.get("connection_id"))
            client = session.get(Client, payload.get("client_id"))
            if not connection or not client:
                return
            process_inbound(session, client, connection, payload.get("message", {}))
        elif job.job_type == "poll_connection":
            connection = session.get(MailboxConnection, payload.get("connection_id"))
            if not connection:
                return
            provider_impl = GmailProvider() if connection.provider == Provider.gmail else MicrosoftProvider()
            tokens = {
                "access_token": decrypt(connection.access_token_encrypted),
                "refresh_token": decrypt(connection.refresh_token_encrypted) if connection.refresh_token_encrypted else None,
            }
            metadata = json.loads(connection.provider_metadata or "{}")
            failure_count = metadata.get("failure_count", 0)
            try:
                changes = provider_impl.list_recent_changes(tokens, metadata)
                metadata["failure_count"] = 0
            except Exception:
                failure_count += 1
                metadata["failure_count"] = failure_count
                if failure_count >= 3:
                    connection.status = ConnectionStatus.needs_reconnect
                    backoff = min(300, 30 * (2 ** (failure_count - 3)))  # exponential up to 5 minutes
                    time.sleep(backoff)
                connection.last_sync_at = datetime.utcnow()
                connection.provider_metadata = json.dumps(metadata)
                session.add(connection)
                session.commit()
                return
            if changes.get("reset_watch"):
                metadata = provider_impl.ensure_inbound_subscription(tokens, metadata)
            if changes.get("history_id"):
                metadata["history_id"] = changes["history_id"]
            if changes.get("delta_link"):
                metadata["delta_link"] = changes["delta_link"]
            connection.provider_metadata = json.dumps(metadata)
            connection.last_sync_at = datetime.utcnow()
            session.add(connection)
            session.commit()
            for msg in changes.get("messages", []):
                enqueue_payload = {
                    "client_id": connection.client_id,
                    "connection_id": connection.id,
                    "message": msg,
                }
                enqueue_job(session, "process_inbound", enqueue_payload)
        elif job.job_type == "renew_watch":
            connection = session.get(MailboxConnection, payload.get("connection_id"))
            if not connection:
                return
            provider_impl = GmailProvider() if connection.provider == Provider.gmail else MicrosoftProvider()
            tokens = {
                "access_token": decrypt(connection.access_token_encrypted),
                "refresh_token": decrypt(connection.refresh_token_encrypted) if connection.refresh_token_encrypted else None,
            }
            metadata = json.loads(connection.provider_metadata or "{}")
            try:
                metadata = provider_impl.ensure_inbound_subscription(tokens, metadata)
                connection.status = ConnectionStatus.active
                metadata["failure_count"] = 0
            except Exception:
                connection.status = ConnectionStatus.needs_reconnect
                metadata["failure_count"] = metadata.get("failure_count", 0) + 1
            connection.provider_metadata = json.dumps(metadata)
            connection.last_sync_at = datetime.utcnow()
            session.add(connection)
            session.commit()
        else:
            raise ValueError(f"Unknown job type {job.job_type}")


def main():
    worker_loop(Processor())


if __name__ == "__main__":
    main()
