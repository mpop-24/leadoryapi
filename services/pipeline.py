from __future__ import annotations

import secrets
import time
from typing import Optional

from sqlmodel import Session, select

from models import (
    AIDraft,
    AIStatus,
    Client,
    Direction,
    EmailMessage,
    EmailThread,
    Lead,
    LeadSource,
    Provider,
    StatusGeneric,
)
from datetime import datetime, timedelta
import json

from providers.gmail import GmailProvider
from providers.microsoft import MicrosoftProvider
from services import ai as ai_service
from services.crypto import decrypt, encrypt
from models import ConnectionStatus
from services.slack import get_client_webhook, send_slack


def get_provider(provider: Provider):
    if provider == Provider.gmail:
        return GmailProvider()
    if provider == Provider.microsoft:
        return MicrosoftProvider()
    raise RuntimeError("Unsupported provider")


def log_event(correlation_id: str, event: str, **fields) -> None:
    payload = {"correlation_id": correlation_id, "event": event}
    payload.update(fields)
    try:
        print(json.dumps(payload, separators=(",", ":"), default=str), flush=True)
    except Exception:
        print(payload, flush=True)


def resolve_tokens(session: Session, connection) -> dict:
    tokens = {
        "access_token": decrypt(connection.access_token_encrypted),
        "refresh_token": decrypt(connection.refresh_token_encrypted) if connection.refresh_token_encrypted else None,
        "expires_at": connection.expires_at,
    }
    provider_impl = get_provider(connection.provider)
    # refresh if expired or about to expire in 2 minutes
    if connection.expires_at and connection.expires_at <= datetime.utcnow() + timedelta(minutes=2):
        try:
            refreshed = provider_impl.refresh_tokens_if_needed(tokens)
            if refreshed.get("access_token"):
                tokens["access_token"] = refreshed["access_token"]
                connection.access_token_encrypted = encrypt(refreshed["access_token"])
            if refreshed.get("refresh_token"):
                tokens["refresh_token"] = refreshed["refresh_token"]
                connection.refresh_token_encrypted = encrypt(refreshed["refresh_token"])
            if refreshed.get("expires_in"):
                connection.expires_at = datetime.utcnow() + timedelta(seconds=int(refreshed["expires_in"]))
            session.add(connection)
            session.commit()
        except Exception as exc:
            connection.status = ConnectionStatus.needs_reconnect
            connection.last_sync_at = datetime.utcnow()
            session.add(connection)
            session.commit()
            raise
    return tokens


def log(msg: str) -> None:
    print(msg, flush=True)


def upsert_thread(session: Session, client_id: int, connection_id: Optional[int], provider: Provider, provider_thread_id: str, subject: str, last_message_at) -> EmailThread:
    stmt = select(EmailThread).where(
        EmailThread.client_id == client_id,
        EmailThread.provider == provider,
        EmailThread.provider_thread_id == provider_thread_id,
    )
    thread = session.exec(stmt).first()
    if thread:
        thread.subject = thread.subject or subject
        thread.last_message_at = last_message_at or thread.last_message_at
    else:
        thread = EmailThread(
            client_id=client_id,
            connection_id=connection_id,
            provider=provider,
            provider_thread_id=provider_thread_id,
            subject=subject,
            last_message_at=last_message_at,
        )
        session.add(thread)
        session.flush()
    return thread


def upsert_message(session: Session, client_id: int, thread_id: Optional[int], provider: Provider, provider_message_id: str, direction: Direction, from_email: str, to_email: str, subject: str, body: str, snippet: str, received_at) -> EmailMessage:
    stmt = select(EmailMessage).where(
        EmailMessage.client_id == client_id,
        EmailMessage.provider == provider,
        EmailMessage.provider_message_id == provider_message_id,
    )
    existing = session.exec(stmt).first()
    if existing:
        return existing
    msg = EmailMessage(
        client_id=client_id,
        thread_id=thread_id,
        provider=provider,
        provider_message_id=provider_message_id,
        direction=direction,
        from_email=from_email,
        to_email=to_email,
        subject=subject,
        body_text=body,
        snippet=snippet,
        received_at=received_at,
    )
    session.add(msg)
    session.flush()
    return msg


def create_lead_if_needed(session: Session, client: Client, thread: EmailThread, message: EmailMessage, correlation_id: str) -> Optional[Lead]:
    # Simple heuristic: create a lead for every inbound not from the business's own connected email
    if message.from_email and client.inquiry_email and message.from_email.lower() == client.inquiry_email.lower():
        return None
    lead = Lead(
        client_id=client.id,
        thread_id=thread.id,
        lead_name=message.from_email,
        lead_email=message.from_email,
        message_text=message.body_text,
        source=LeadSource.email_inbox,
        correlation_id=correlation_id,
    )
    session.add(lead)
    session.flush()
    return lead


def notify_slack(client: Client, lead: Lead, message: EmailMessage) -> None:
    webhook = get_client_webhook(client.slack_webhook_url)
    if not webhook:
        return
    text = (
        f"New lead for {client.business_name}\n"
        f"From: {message.from_email}\n"
        f"Subject: {message.subject or ''}\n"
        f"Snippet: {message.snippet or ''}"
    )
    ok, err = send_slack(webhook, text)
    lead.slack_status = StatusGeneric.sent if ok else StatusGeneric.failed
    if err:
        lead.error_slack = err
    log_event(lead.correlation_id or "", "SLACK_SEND_RESULT", status=lead.slack_status, error=lead.error_slack)


def draft_and_send(session: Session, client: Client, connection, message: EmailMessage, thread: EmailThread, lead: Lead) -> None:
    correlation_id = lead.correlation_id or secrets.token_hex(8)
    if ai_service.is_spam_or_unsafe(message.body_text or ""):
        lead.ai_status = AIStatus.failed
        lead.error_ai = "spam_or_unsafe"
        lead.email_status = StatusGeneric.failed
        log_event(correlation_id, "AI_DRAFT_RESULT", status="failed", error="spam_or_unsafe")
        return

    draft_text = ai_service.generate_draft(
        text=message.body_text or "",
        business_name=client.business_name,
        business_description=client.business_description,
        pricing=client.pricing,
        sign_off_name=client.sign_off_name,
        mimic_email=client.mimic_email,
    )
    if not draft_text:
        lead.ai_status = AIStatus.failed
        lead.error_ai = "ai_generation_failed"
        log_event(correlation_id, "AI_DRAFT_RESULT", status="failed", error="ai_generation_failed")
        return

    draft = AIDraft(
        lead_id=lead.id,
        prompt_text="",
        draft_text=draft_text,
        model_name=ai_service.DEFAULT_MODEL,
    )
    session.add(draft)
    session.flush()
    lead.ai_status = AIStatus.drafted
    log_event(correlation_id, "AI_DRAFT_RESULT", status="drafted", lead_id=lead.id)

    provider_impl = get_provider(connection.provider)
    tokens = resolve_tokens(session, connection)
    subject = sanitize_header(f"Re: {message.subject}") if message.subject else "Re: Your inquiry"
    body = draft_text
    if client.sign_off_name:
        body = f"{body}\n\n{client.sign_off_name}"

    attempt = 0
    max_attempts = 3
    backoff = 2
    while attempt < max_attempts:
        attempt += 1
        try:
            log_event(correlation_id, "EMAIL_SEND_ATTEMPT", attempt=attempt, to=message.from_email)
            provider_impl.send_message(
                tokens=tokens,
                thread_id=thread.provider_thread_id,
                to=sanitize_header(message.from_email or ""),
                subject=subject,
                body=body,
                reply_to_message_id=message.provider_message_id,
            )
            lead.email_status = StatusGeneric.sent
            log_event(correlation_id, "EMAIL_SEND_RESULT", status="sent", attempt=attempt)
            return
        except Exception as exc:  # pragma: no cover - provider failure path
            lead.error_email = str(exc)
            log_event(correlation_id, "EMAIL_SEND_RESULT", status="failed", attempt=attempt, error=str(exc))
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
    lead.email_status = StatusGeneric.failed


def _ensure_full_message(session: Session, connection, provider_message: dict, correlation_id: str) -> dict:
    # If body/subject missing, fetch from provider using tokens
    if provider_message.get("body") and provider_message.get("subject"):
        return provider_message
    provider_impl = get_provider(connection.provider)
    tokens = resolve_tokens(session, connection)
    fetched = provider_impl.fetch_message(tokens, provider_message["id"])
    for k, v in fetched.items():
        provider_message.setdefault(k, v)
    log_event(correlation_id, "MESSAGE_FETCHED", provider=connection.provider, message_id=provider_message.get("id"))
    return provider_message


def process_inbound(session: Session, client: Client, connection, provider_message: dict) -> None:
    correlation_id = secrets.token_hex(8)
    log_event(correlation_id, "INBOUND_NOTIFICATION_RECEIVED", provider=connection.provider, connection_id=connection.id)
    fetched = _ensure_full_message(session, connection, provider_message, correlation_id)
    from_email = sanitize_header(fetched.get("from_email") or fetched.get("from") or "unknown")
    to_email = sanitize_header(fetched.get("to_email") or connection.connected_email or "")
    subject_clean = sanitize_header(fetched.get("subject", ""))
    thread = upsert_thread(
        session,
        client_id=client.id,
        connection_id=connection.id,
        provider=connection.provider,
        provider_thread_id=fetched.get("thread_id") or fetched.get("id"),
        subject=subject_clean,
        last_message_at=fetched.get("received_at"),
    )
    log_event(correlation_id, "THREAD_UPSERTED", thread_id=thread.id, provider_thread_id=thread.provider_thread_id)
    message = upsert_message(
        session,
        client_id=client.id,
        thread_id=thread.id,
        provider=connection.provider,
        provider_message_id=fetched.get("id"),
        direction=Direction.inbound,
        from_email=from_email,
        to_email=to_email,
        subject=subject_clean,
        body=fetched.get("body", "") or "",
        snippet=fetched.get("snippet", "") or "",
        received_at=fetched.get("received_at"),
    )
    log_event(correlation_id, "MESSAGE_UPSERTED", message_id=message.id, provider_message_id=message.provider_message_id)
    lead = create_lead_if_needed(session, client, thread, message, correlation_id)
    if lead:
        log_event(correlation_id, "LEAD_CREATED", lead_id=lead.id, client_id=client.id)
        notify_slack(client, lead, message)
        draft_and_send(session, client, connection, message, thread, lead)
    session.commit()
def sanitize_header(value: str) -> str:
    # remove CR/LF to avoid header injection
    return value.replace("\r", " ").replace("\n", " ").strip()
