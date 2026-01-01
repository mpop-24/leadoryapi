from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from db import get_db
from models import Client, ConnectionStatus, MailboxConnection, Provider
from providers.gmail import GmailProvider
from providers.microsoft import MicrosoftProvider
from services.crypto import encrypt
from routers.admin import require_admin
from services.security import sign_state, verify_state

router = APIRouter(tags=["oauth"])


def get_provider(provider: Provider):
    if provider == Provider.gmail:
        return GmailProvider()
    if provider == Provider.microsoft:
        return MicrosoftProvider()
    raise HTTPException(status_code=400, detail="Unsupported provider")


@router.post("/admin/clients/{client_id}/connect/google")
async def connect_google(client_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    state = sign_state(client_id, Provider.gmail.value)
    provider = get_provider(Provider.gmail)
    return {"auth_url": provider.get_authorization_url(state)}


@router.post("/admin/clients/{client_id}/connect/microsoft")
async def connect_microsoft(client_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    state = sign_state(client_id, Provider.microsoft.value)
    provider = get_provider(Provider.microsoft)
    return {"auth_url": provider.get_authorization_url(state)}


@router.get("/oauth/google/callback")
async def google_callback(code: str = Query(...), state: str = Query(...), session: Session = Depends(get_db)):
    verified = verify_state(state)
    if not verified:
        raise HTTPException(status_code=400, detail="Invalid state")
    client_id, provider_value = verified
    if provider_value != Provider.gmail.value:
        raise HTTPException(status_code=400, detail="Invalid state")
    provider = get_provider(Provider.gmail)
    tokens = provider.exchange_code_for_tokens(code, provider.redirect_uri)
    expires_at = datetime.utcnow() + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    metadata = provider.ensure_inbound_subscription({"access_token": tokens["access_token"]}, {})
    connection = MailboxConnection(
        client_id=client_id,
        provider=Provider.gmail,
        connected_email=tokens.get("email") or tokens.get("id_token", "unknown"),
        access_token_encrypted=encrypt(tokens["access_token"]),
        refresh_token_encrypted=encrypt(tokens.get("refresh_token", "")) if tokens.get("refresh_token") else None,
        status=ConnectionStatus.active,
        expires_at=expires_at,
        provider_metadata=json.dumps(metadata),
    )
    session.add(connection)
    session.commit()
    session.refresh(connection)
    return {"ok": True, "connection_id": connection.id}


@router.get("/oauth/microsoft/callback")
async def microsoft_callback(code: str = Query(...), state: str = Query(...), session: Session = Depends(get_db)):
    verified = verify_state(state)
    if not verified:
        raise HTTPException(status_code=400, detail="Invalid state")
    client_id, provider_value = verified
    if provider_value != Provider.microsoft.value:
        raise HTTPException(status_code=400, detail="Invalid state")
    provider = get_provider(Provider.microsoft)
    tokens = provider.exchange_code_for_tokens(code, provider.redirect_uri)
    expires_at = datetime.utcnow() + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    metadata = provider.ensure_inbound_subscription({"access_token": tokens["access_token"]}, {})
    connection = MailboxConnection(
        client_id=client_id,
        provider=Provider.microsoft,
        connected_email=tokens.get("id_token", "unknown"),
        access_token_encrypted=encrypt(tokens["access_token"]),
        refresh_token_encrypted=encrypt(tokens.get("refresh_token", "")) if tokens.get("refresh_token") else None,
        status=ConnectionStatus.active,
        expires_at=expires_at,
        provider_metadata=json.dumps(metadata),
    )
    session.add(connection)
    session.commit()
    session.refresh(connection)
    return {"ok": True, "connection_id": connection.id}


@router.get("/admin/clients/{client_id}/connection-status")
async def connection_status(client_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    connections = session.exec(select(MailboxConnection).where(MailboxConnection.client_id == client_id)).all()
    if not connections:
        return {"status": "none"}
    conn = connections[0]
    return {
        "status": conn.status,
        "provider": conn.provider,
        "connected_email": conn.connected_email,
        "last_sync_at": conn.last_sync_at,
    }


@router.post("/admin/clients/{client_id}/disconnect")
async def disconnect(client_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    connections = session.exec(select(MailboxConnection).where(MailboxConnection.client_id == client_id)).all()
    for c in connections:
        c.status = ConnectionStatus.revoked
        session.add(c)
    session.commit()
    return {"ok": True}
