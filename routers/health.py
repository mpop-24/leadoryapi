from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from sqlmodel import Session
from sqlalchemy import text

from db import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/health/db")
async def health_db(session: Session = Depends(get_db)):
    try:
        session.exec(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


@router.get("/health/providers")
async def health_providers():
    return {
        "google": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")),
        "microsoft": bool(os.getenv("MICROSOFT_CLIENT_ID") and os.getenv("MICROSOFT_CLIENT_SECRET")),
    }
