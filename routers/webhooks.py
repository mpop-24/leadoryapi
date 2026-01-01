from __future__ import annotations

import os
import base64
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from db import get_db
from models import Client, MailboxConnection, Provider
from services.queue import enqueue_job
from providers.gmail import GmailProvider

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _find_gmail_connection(session: Session, email_address: str | None, topic_id: str | None, connection_id: str | None) -> MailboxConnection | None:
    connection = None
    if email_address:
        stmt = select(MailboxConnection).where(
            MailboxConnection.provider == Provider.gmail, MailboxConnection.connected_email == email_address
        )
        connection = session.exec(stmt).first()
    if not connection and connection_id:
        connection = session.get(MailboxConnection, connection_id)
    if not connection and topic_id:
        gmail_conns = session.exec(select(MailboxConnection).where(MailboxConnection.provider == Provider.gmail)).all()
        matches = []
        email_matches = []
        for conn in gmail_conns:
            try:
                meta = json.loads(conn.provider_metadata or "{}")
            except Exception:
                meta = {}
            if meta.get("topic_name") == topic_id:
                matches.append(conn)
                if email_address and conn.connected_email == email_address:
                    email_matches.append(conn)
        pick_from = email_matches if email_matches else matches
        if len(pick_from) == 1:
            connection = pick_from[0]
        elif len(pick_from) > 1:
            connection = sorted(
                pick_from,
                key=lambda c: json.loads(c.provider_metadata or "{}").get("watch_created_at", ""),
                reverse=True,
            )[0]
    return connection


@router.post("/gmail/push")
async def gmail_push(request: Request, session: Session = Depends(get_db)):
    auth_header = request.headers.get("authorization", "")
    secret = request.headers.get("x-gmail-secret") or request.query_params.get("secret")
    expected_secret = os.getenv("GMAIL_WEBHOOK_SECRET", "")
    provider_impl = GmailProvider()
    if expected_secret and secret != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if auth_header and not provider_impl.verify_pubsub_jwt(auth_header):
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    message = body.get("message", {})
    data_b64 = message.get("data", "")
    email_address = None
    history_id = None
    if data_b64:
        try:
            # Google may omit padding
            padding = "=" * (-len(data_b64) % 4)
            decoded = base64.b64decode(data_b64 + padding).decode("utf-8")
            parsed = json.loads(decoded)
            email_address = parsed.get("emailAddress")
            history_id = parsed.get("historyId")
        except Exception:
            pass
    topic_id = (message.get("attributes") or {}).get("topicId") or (message.get("attributes") or {}).get("topic_id")
    connection_id = body.get("connection_id")
    connection = _find_gmail_connection(session, email_address, topic_id, connection_id)
    if not connection or connection.provider != Provider.gmail:
        raise HTTPException(status_code=404, detail="connection not found")
    client = session.get(Client, connection.client_id)
    if not client:
        raise HTTPException(status_code=404, detail="client not found")
    # Update history_id in metadata for polling
    try:
        meta = json.loads(connection.provider_metadata or "{}")
    except Exception:
        meta = {}
    if history_id:
        meta["history_id"] = history_id
    if topic_id:
        meta["topic_name"] = topic_id
    connection.provider_metadata = json.dumps(meta)
    session.add(connection)
    session.commit()
    # Enqueue a poll to fetch new messages since history_id
    enqueue_job(session, "poll_connection", {"connection_id": connection.id})
    return {"ok": True}


@router.post("/microsoft/notifications")
async def ms_notifications(request: Request, validationToken: str | None = None, session: Session = Depends(get_db)):
    if validationToken:
        return validationToken  # Graph handshake
    body = await request.json()
    notifications = body.get("value", [])
    secret = os.getenv("MICROSOFT_SUBSCRIPTION_CLIENT_STATE_SECRET", "")
    if secret and not notifications:
        raise HTTPException(status_code=401, detail="Unauthorized")
    processed = 0
    for n in notifications:
        if secret and n.get("clientState") != secret:
            continue
        subscription_id = n.get("subscriptionId")
        resource_data = n.get("resourceData", {})
        message_id = resource_data.get("id")
        if not subscription_id or not message_id:
            continue
        from_email = (resource_data.get("from") or {}).get("emailAddress", {}).get("address")
        to_email = connection.connected_email
        if resource_data.get("toRecipients"):
            to_email = (resource_data.get("toRecipients")[0].get("emailAddress") or {}).get("address")
        subject = resource_data.get("subject")
        body = resource_data.get("bodyPreview")
        snippet = resource_data.get("bodyPreview")
        received_at = resource_data.get("receivedDateTime")
        connection = None
        ms_conns = session.exec(select(MailboxConnection).where(MailboxConnection.provider == Provider.microsoft)).all()
        for conn in ms_conns:
            try:
                meta = json.loads(conn.provider_metadata or "{}")
            except Exception:
                meta = {}
            if meta.get("subscription_id") == subscription_id:
                connection = conn
                break
        if not connection:
            continue
        client = session.get(Client, connection.client_id)
        if not client:
            continue
        enqueue_job(
            session,
            "process_inbound",
            {
                "client_id": client.id,
                "connection_id": connection.id,
                "message": {
                    "id": message_id,
                    "thread_id": resource_data.get("conversationId"),
                    "from_email": from_email,
                    "to_email": to_email,
                    "subject": subject,
                    "body": body,
                    "snippet": snippet,
                    "received_at": received_at,
                },
            },
        )
        processed += 1
    if secret and processed == 0:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"ok": True, "processed": processed}
