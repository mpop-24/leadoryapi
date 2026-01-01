from __future__ import annotations

import hmac
import os
import time
from typing import List

import jwt
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlmodel import Session, select

from db import get_db
from models import Client, ReplyMode

router = APIRouter(prefix="/admin", tags=["admin"])

JWT_ALG = "HS256"


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET missing")
    return secret


def create_token(sub: str) -> str:
    payload = {"sub": sub, "exp": int(time.time()) + 86400}
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALG)


def require_admin(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALG])
        return str(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


class LoginPayload(BaseModel):
    username: str
    password: str


@router.post("/login")
async def admin_login(payload: LoginPayload):
    admin_user = os.getenv("ADMIN_USERNAME", "")
    admin_pass = os.getenv("ADMIN_PASSWORD", "")
    if not admin_user or not admin_pass:
        raise HTTPException(status_code=500, detail="Admin creds missing")
    if not (hmac.compare_digest(payload.username, admin_user) and hmac.compare_digest(payload.password, admin_pass)):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(admin_user), "token_type": "bearer"}


class ClientCreate(BaseModel):
    slug: str
    business_name: str
    inquiry_email: str | None = None
    pricing: str | None = None
    business_description: str | None = None
    sign_off_name: str | None = None
    mimic_email: str | None = None
    slack_webhook_url: str | None = None
    reply_mode: ReplyMode = ReplyMode.auto_send


class ClientUpdate(BaseModel):
    slug: str | None = None
    business_name: str | None = None
    inquiry_email: str | None = None
    pricing: str | None = None
    business_description: str | None = None
    sign_off_name: str | None = None
    mimic_email: str | None = None
    slack_webhook_url: str | None = None
    reply_mode: ReplyMode | None = None


@router.get("/clients", response_model=List[Client])
async def list_clients(_: str = Depends(require_admin), session: Session = Depends(get_db)):
    return session.exec(select(Client).order_by(Client.created_at.desc())).all()


@router.post("/clients", response_model=Client)
async def create_client(payload: ClientCreate, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    existing = session.exec(select(Client).where(Client.slug == payload.slug)).first()
    if existing:
        raise HTTPException(status_code=400, detail="slug exists")
    client = Client(**payload.model_dump())
    session.add(client)
    session.commit()
    session.refresh(client)
    return client


@router.get("/clients/{client_id}", response_model=Client)
async def get_client(client_id: int, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Not found")
    return client


@router.patch("/clients/{client_id}", response_model=Client)
async def patch_client(client_id: int, payload: ClientUpdate, _: str = Depends(require_admin), session: Session = Depends(get_db)):
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(client, k, v)
    session.add(client)
    session.commit()
    session.refresh(client)
    return client
