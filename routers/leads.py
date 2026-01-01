from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from db import get_db
from models import Client, Lead, LeadSource, MailboxConnection
from routers.admin import require_admin
from services.queue import enqueue_job

router = APIRouter(tags=["leads"])


@router.get("/admin/leads")
async def list_leads(limit: int = 50, client_id: int | None = None, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    stmt = select(Lead).order_by(Lead.created_at.desc()).limit(limit)
    if client_id:
        stmt = stmt.where(Lead.client_id == client_id)
    return session.exec(stmt).all()


@router.get("/admin/leads/{lead_id}")
async def get_lead(lead_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    lead = session.get(Lead, lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Not found")
    return lead


@router.post("/lead/{company_slug}")
async def submit_lead(company_slug: str, request: Request, session: Session = Depends(get_db)):
    data = await request.json()
    client = session.exec(select(Client).where(Client.slug == company_slug)).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    connection = session.exec(select(MailboxConnection).where(MailboxConnection.client_id == client.id)).first()
    if not connection:
        raise HTTPException(status_code=400, detail="No mailbox connected")
    fake_provider_message = {
        "thread_id": data.get("thread_id") or data.get("email"),
        "id": (data.get("email") or "") + "-form",
        "from_email": data.get("email"),
        "to_email": connection.connected_email,
        "subject": data.get("subject") or "New inquiry",
        "body": data.get("message") or "",
        "snippet": (data.get("message") or "")[:100],
        "received_at": None,
    }
    enqueue_job(
        session,
        "process_inbound",
        {"client_id": client.id, "connection_id": connection.id, "message": fake_provider_message},
    )
    return {"ok": True, "enqueued": True}
