from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
import threading
from datetime import datetime, timezone
from queue import Queue
from typing import Any, List, Optional

import httpx
import jwt
from fastapi import (
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import EmailStr
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import UniqueConstraint, inspect, text

RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SEC = 60
rate_limits: dict[str, List[float]] = {}

JWT_ALGORITHM = "HS256"
job_queue: "Queue[int]" = Queue()
worker_thread: Optional[threading.Thread] = None


# ------------------------------------------------------------------------------
# DB setup
# ------------------------------------------------------------------------------
def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return normalize_database_url(database_url)

    fallback = "sqlite:///./leadory.db"
    print("WARNING: DATABASE_URL is not set; using local sqlite database for dev.")
    return fallback


engine = create_engine(get_database_url(), echo=False)


def get_session() -> Session:
    with Session(engine) as session:
        yield session


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_schema() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "client" in table_names:
        client_columns = {col["name"] for col in inspector.get_columns("client")}
        missing_client_columns = []
        if "pricing" not in client_columns:
            missing_client_columns.append("ADD COLUMN pricing TEXT")
        if "business_description" not in client_columns:
            missing_client_columns.append("ADD COLUMN business_description TEXT")
        if "sign_off_name" not in client_columns:
            missing_client_columns.append("ADD COLUMN sign_off_name VARCHAR(255)")
        if "mimic_email" not in client_columns:
            missing_client_columns.append("ADD COLUMN mimic_email VARCHAR(255)")
        if "inquiry_email" not in client_columns:
            missing_client_columns.append("ADD COLUMN inquiry_email VARCHAR(255)")

        if missing_client_columns:
            with engine.begin() as conn:
                for clause in missing_client_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE client {clause}"))
                    except Exception as exc:
                        print(f"Schema migration warning (client): {clause}: {exc}")


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column_kwargs={"onupdate": utcnow}
    )


class Client(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    company_slug: str = Field(index=True, unique=True)
    is_active: bool = Field(default=True, index=True)
    inbound_address: str = Field(unique=True, index=True)
    from_email: str
    inquiry_email: Optional[EmailStr] = Field(default=None, index=True)
    reply_to_email: Optional[str] = None
    cc_email: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    pricing: Optional[str] = None
    business_description: Optional[str] = None
    sign_off_name: Optional[str] = Field(default=None, max_length=255)
    mimic_email: Optional[EmailStr] = Field(default=None, index=True)


class InboundEmail(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    provider: str = Field(default="sendgrid", index=True)
    provider_message_id: str = Field(index=True)
    from_email: str
    to_email: str
    subject: Optional[str] = None
    body_text: Optional[str] = None
    received_at: datetime = Field(default_factory=utcnow)
    classification: Optional[str] = Field(default=None, index=True)
    reply_status: Optional[str] = Field(default="RECEIVED", index=True)
    replied_at: Optional[datetime] = None
    error: Optional[str] = None

    __table_args__ = (
        UniqueConstraint("client_id", "provider_message_id", name="uq_client_message"),
    )


class Lead(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    name: Optional[str] = None
    email: Optional[EmailStr] = Field(default=None, index=True)
    phone: Optional[str] = None
    message_text: Optional[str] = None
    meta_json: Optional[str] = None
    email_status: str = Field(default="pending", index=True)
    slack_status: str = Field(default="pending", index=True)
    error_email: Optional[str] = None
    error_slack: Optional[str] = None

# ------------------------------------------------------------------------------
# App init
# ------------------------------------------------------------------------------
app = FastAPI(title="Leadory Email Engine", version="0.2.0", redirect_slashes=False)

origins = [
    "https://leadory.org",
    "https://www.leadory.org",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)
    ensure_schema()
    start_worker()


# ------------------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------------------
def get_admin_key() -> str | None:
    return os.getenv("ADMIN_KEY")


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured.",
        )
    return secret


def get_jwt_expires_seconds() -> int:
    try:
        return int(os.getenv("JWT_EXPIRES_SECONDS", "86400"))
    except ValueError:
        return 86400


def create_access_token(username: str) -> tuple[str, int]:
    expires_in = get_jwt_expires_seconds()
    payload = {
        "sub": username,
        "exp": int(time.time()) + expires_in,
    }
    token = jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)
    return token, expires_in


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        ) from exc

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return str(username)


def get_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def require_admin(request: Request) -> str:
    token = get_bearer_token(request)
    admin_key = get_admin_key()
    header_key = request.headers.get("x-admin-key", "")
    token_user: Optional[str] = None

    if token:
        token_user = decode_access_token(token)
    elif admin_key and hmac.compare_digest(header_key, admin_key):
        token_user = "admin"

    if not token_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return token_user


def rate_limit(ip: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    hits = rate_limits.get(ip, [])
    hits = [t for t in hits if t >= window_start]
    if len(hits) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded, please retry shortly.",
        )
    hits.append(now)
    rate_limits[ip] = hits


def log_event(correlation_id: str, event: str, **fields: Any) -> None:
    payload = {"correlation_id": correlation_id, "event": event}
    payload.update(fields)
    try:
        print(json.dumps(payload, separators=(",", ":"), default=str))
    except Exception:
        print({"correlation_id": correlation_id, "event": event, "fields": fields})


def sanitize_header(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def resolve_from_email() -> Optional[str]:
    for env_key in ["FROM_EMAIL", "SENDGRID_FROM_EMAIL", "VERIFIED_SENDER_EMAIL"]:
        candidate = os.getenv(env_key, "").strip()
        if candidate:
            return sanitize_header(candidate)
    return None


# ------------------------------------------------------------------------------
# SendGrid / AI helpers
# ------------------------------------------------------------------------------
def inbound_secret() -> str:
    secret = os.getenv("INBOUND_WEBHOOK_SECRET", "")
    if not secret:
        raise RuntimeError("INBOUND_WEBHOOK_SECRET is required.")
    return secret


def inbound_domain() -> str:
    return os.getenv("INBOUND_EMAIL_DOMAIN", "inbound.leadory.co")


def parse_message_id(headers_raw: str) -> Optional[str]:
    match = re.search(r"Message-ID:\s*<?([^>\s]+)>?", headers_raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def compute_provider_message_id(to_email: str, from_email: str, subject: str, text: str) -> str:
    base = f"{to_email}|{from_email}|{subject}|{text[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def is_hard_filtered(text: str) -> bool:
    lowered = text.lower()
    banned_keywords = ["unsubscribe", "newsletter", "receipt", "invoice", "order confirmation"]
    if not text.strip():
        return True
    if len(text.strip()) < 10:
        return True
    return any(keyword in lowered for keyword in banned_keywords)


def classify_lead(text: str) -> str:
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        return "ERROR"
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "Classify if this email is a real lead asking for service. "
            "Answer strictly with YES or NO.\n\n"
            f"Email:\n{text}\n"
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = completion.choices[0].message.content.strip().upper()
        return "YES" if reply.startswith("YES") else "NO"
    except Exception:
        return "ERROR"


def generate_reply(text: str, client: Client) -> str:
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        return "Hi there, thanks for reaching out! A team member will respond shortly."
    try:
        import openai

        client_ai = openai.OpenAI(api_key=api_key)
        prompt = (
            "You are composing a concise, friendly response to a new inbound lead. "
            f"Client name: {client.name}. "
            "Write 3-6 sentences, ask a clarifying question, and keep it professional.\n\n"
            f"Inbound message:\n{text}\n"
        )
        completion = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "Thanks for your message! A team member will follow up shortly."


def send_slack_notification(webhook_url: str, message: str) -> None:
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(webhook_url, json={"text": message})
        if resp.status_code >= 300:
            raise RuntimeError(f"Slack response {resp.status_code}: {resp.text}")


def send_sendgrid_email(
    client: Client,
    inbound: InboundEmail,
    body: str,
) -> None:
    api_key = os.getenv("SENDGRID_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SENDGRID_API_KEY is required to send email.")

    from_email = resolve_from_email() or client.from_email
    if not from_email:
        raise RuntimeError("MISSING_FROM_EMAIL")
    from_email = sanitize_header(from_email)
    to_email = sanitize_header(inbound.from_email)
    subject = (
        sanitize_header(f"Re: {inbound.subject}") if inbound.subject else "Re: Your inquiry"
    )

    if from_email.endswith(f"@{inbound_domain()}") and not client.reply_to_email:
        raise RuntimeError("reply_to_email is required when using Leadory sending domain.")

    payload: dict[str, Any] = {
        "personalizations": [
            {
                "to": [{"email": to_email}],
                "subject": subject,
            }
        ],
        "from": {"email": from_email},
        "content": [{"type": "text/plain", "value": body}],
        "headers": {
            "X-Leadory-Client-ID": str(client.id),
            "X-Leadory-Inbound-ID": str(inbound.id),
        },
    }

    if client.cc_email:
        payload["personalizations"][0]["cc"] = [{"email": client.cc_email}]
    if client.reply_to_email:
        payload["reply_to"] = {"email": client.reply_to_email}

    attempts = 0
    backoff = 1
    last_error: Optional[str] = None

    while attempts < 3:
        attempts += 1
        try:
            with httpx.Client(timeout=10.0) as client_http:
                resp = client_http.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(payload),
                )
                if resp.status_code < 300:
                    return
                last_error = f"SendGrid status {resp.status_code}: {resp.text}"
        except Exception as exc:  # pragma: no cover - safety net
            last_error = str(exc)
        time.sleep(backoff)
        backoff *= 2

    raise RuntimeError(last_error or "Failed to send email via SendGrid.")


def process_inbound_email(inbound_id: int) -> None:
    with Session(engine) as session:
        inbound = session.get(InboundEmail, inbound_id)
        if not inbound:
            return
        client = session.get(Client, inbound.client_id)
        if not client or not client.is_active:
            inbound.classification = "NO"
            inbound.reply_status = "SKIPPED"
            inbound.error = "Client inactive or missing."
            session.add(inbound)
            session.commit()
            return

        text = inbound.body_text or ""
        if is_hard_filtered(text):
            inbound.classification = "NO"
            inbound.reply_status = "SKIPPED"
            session.add(inbound)
            session.commit()
            _notify_slack_status(client, inbound)
            return

        classification = classify_lead(text)
        inbound.classification = classification

        if classification != "YES":
            inbound.reply_status = "SKIPPED"
            if classification == "ERROR":
                inbound.error = "AI classification failed"
            session.add(inbound)
            session.commit()
            _notify_slack_status(client, inbound)
            return

        try:
            reply_body = generate_reply(text, client)
            send_sendgrid_email(client, inbound, reply_body)
            inbound.reply_status = "SENT"
            inbound.replied_at = utcnow()
            session.add(inbound)
            session.commit()
        except Exception as exc:
            inbound.reply_status = "FAILED"
            inbound.error = str(exc)
            session.add(inbound)
            session.commit()
            _notify_slack_status(client, inbound, failed=True)


def _notify_slack_status(client: Client, inbound: InboundEmail, failed: bool = False) -> None:
    if not client.slack_webhook_url:
        return

    status_text = inbound.reply_status or "UNKNOWN"
    title = "FAILED — manual follow-up needed" if failed else f"Lead status: {status_text}"
    snippet = (inbound.body_text or "")[:200]
    message = (
        f"{title}\n"
        f"Client: {client.name}\n"
        f"From: {inbound.from_email}\n"
        f"Subject: {inbound.subject or '(no subject)'}\n"
        f"Body: {snippet}"
    )
    send_slack_notification(client.slack_webhook_url, message)


def _worker_loop() -> None:
    while True:
        inbound_id = job_queue.get()
        try:
            process_inbound_email(inbound_id)
        except Exception as exc:
            print(f"Error processing inbound {inbound_id}: {exc}")
        finally:
            job_queue.task_done()


def start_worker() -> None:
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="inbound-worker")
    worker_thread.start()


def enqueue_inbound_processing(inbound_id: int) -> None:
    job_queue.put_nowait(inbound_id)


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/health/email")
async def health_email() -> JSONResponse:
    api_key = os.getenv("SENDGRID_API_KEY", "").strip()
    from_email = resolve_from_email()
    status_ok = bool(api_key and from_email)
    detail = {
        "sendgrid_key": bool(api_key),
        "from_email": from_email,
    }
    return JSONResponse({"ok": status_ok, **detail})


@app.post("/admin/login")
async def admin_login(
    request: Request,
    payload: dict[str, str] = Body(..., example={"username": "admin", "password": "password"}),
) -> dict[str, Any]:
    ip = request.client.host if request.client else "unknown"
    rate_limit(ip)

    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    admin_username = os.getenv("ADMIN_USERNAME", "")
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if not admin_username or not admin_password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin credentials not configured.",
        )

    if not (
        hmac.compare_digest(username, admin_username)
        and hmac.compare_digest(password, admin_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token, expires_in = create_access_token(admin_username)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


@app.get("/admin/me")
async def admin_me(request: Request) -> dict[str, Any]:
    username = require_admin(request)
    return {"ok": True, "username": username}


class ClientCreate(SQLModel):
    name: str
    company_slug: str
    from_email: str
    inquiry_email: Optional[EmailStr] = None
    reply_to_email: Optional[str] = None
    cc_email: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    pricing: Optional[str] = None
    business_description: Optional[str] = None
    sign_off_name: Optional[str] = None
    mimic_email: Optional[EmailStr] = None
    is_active: bool = True
    inbound_address: Optional[str] = None


class ClientUpdate(SQLModel):
    name: Optional[str] = None
    company_slug: Optional[str] = None
    from_email: Optional[str] = None
    inquiry_email: Optional[EmailStr] = None
    reply_to_email: Optional[str] = None
    cc_email: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    pricing: Optional[str] = None
    business_description: Optional[str] = None
    sign_off_name: Optional[str] = None
    mimic_email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    inbound_address: Optional[str] = None


def generate_inbound_address(client_id: int) -> str:
    return f"client_{client_id}@{inbound_domain()}"


@app.post("/admin/clients", response_model=Client)
async def create_client(
    request: Request,
    payload: ClientCreate,
    session: Session = Depends(get_session),
) -> Client:
    require_admin(request)
    slug = payload.company_slug.strip().lower()
    existing = session.exec(select(Client).where(Client.company_slug == slug)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_slug already exists.",
        )
    inbound_address = payload.inbound_address.strip().lower() if payload.inbound_address else None
    if inbound_address:
        existing_inbound = session.exec(
            select(Client).where(Client.inbound_address == inbound_address)
        ).first()
        if existing_inbound:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="inbound_address already exists.",
            )
    temp_inbound = inbound_address or f"pending-{secrets.token_hex(4)}"

    client = Client(
        name=payload.name.strip(),
        company_slug=slug,
        from_email=payload.from_email.strip(),
        inquiry_email=payload.inquiry_email,
        reply_to_email=payload.reply_to_email.strip() if payload.reply_to_email else None,
        cc_email=payload.cc_email.strip() if payload.cc_email else None,
        slack_webhook_url=payload.slack_webhook_url.strip()
        if payload.slack_webhook_url
        else None,
        pricing=payload.pricing.strip() if payload.pricing else None,
        business_description=payload.business_description.strip()
        if payload.business_description
        else None,
        sign_off_name=payload.sign_off_name.strip() if payload.sign_off_name else None,
        mimic_email=payload.mimic_email,
        is_active=payload.is_active,
        inbound_address=temp_inbound,
    )
    session.add(client)
    session.commit()
    session.refresh(client)

    if not inbound_address:
        client.inbound_address = generate_inbound_address(client.id)
        session.add(client)
        session.commit()
        session.refresh(client)

    return client


@app.put("/admin/clients/{client_id}", response_model=Client)
async def update_client(
    request: Request,
    client_id: int,
    payload: ClientUpdate,
    session: Session = Depends(get_session),
) -> Client:
    require_admin(request)
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")

    if payload.company_slug and payload.company_slug != client.company_slug:
        slug = payload.company_slug.strip().lower()
        existing = session.exec(select(Client).where(Client.company_slug == slug)).first()
        if existing and existing.id != client.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="company_slug already exists.",
            )
        client.company_slug = slug

    for field_name, value in payload.dict(exclude_unset=True).items():
        if field_name == "company_slug":
            continue
        if field_name == "inbound_address" and value:
            value = value.strip().lower()
            existing_inbound = session.exec(
                select(Client).where(Client.inbound_address == value)
            ).first()
            if existing_inbound and existing_inbound.id != client.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="inbound_address already exists.",
                )
        if isinstance(value, str):
            value = value.strip()
        setattr(client, field_name, value)

    session.add(client)
    session.commit()
    session.refresh(client)
    return client


@app.patch("/admin/clients/{client_id}", response_model=Client)
async def patch_client(
    request: Request,
    client_id: int,
    payload: ClientUpdate,
    session: Session = Depends(get_session),
) -> Client:
    return await update_client(request, client_id, payload, session)


@app.get("/admin/clients", response_model=List[Client])
async def list_clients(
    request: Request,
    session: Session = Depends(get_session),
) -> List[Client]:
    require_admin(request)
    clients = session.exec(select(Client).order_by(Client.created_at.desc())).all()
    return clients


@app.get("/admin/clients/{client_id}", response_model=Client)
async def get_client(
    request: Request,
    client_id: int,
    session: Session = Depends(get_session),
) -> Client:
    require_admin(request)
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


@app.post("/admin/clients/{client_id}/toggle")
async def toggle_client(
    request: Request,
    client_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    require_admin(request)
    client = session.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    client.is_active = not client.is_active
    session.add(client)
    session.commit()
    session.refresh(client)
    return {"client_id": client.id, "is_active": client.is_active}


@app.get("/admin/inbound-emails", response_model=List[InboundEmail])
async def list_inbound_emails(
    request: Request,
    session: Session = Depends(get_session),
    client_id: Optional[int] = None,
    limit: int = 50,
) -> List[InboundEmail]:
    require_admin(request)
    stmt = select(InboundEmail).order_by(InboundEmail.received_at.desc()).limit(limit)
    if client_id:
        stmt = stmt.where(InboundEmail.client_id == client_id)
    emails = session.exec(stmt).all()
    return emails


# ------------------------------------------------------------------------------
# Webhook + lead intake
# ------------------------------------------------------------------------------
def validate_inbound_secret(request: Request) -> None:
    provided = request.headers.get("x-inbound-secret") or request.query_params.get("secret")
    if not provided or not hmac.compare_digest(provided, inbound_secret()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.post("/webhooks/sendgrid/inbound")
async def sendgrid_inbound(
    request: Request,
    session: Session = Depends(get_session),
) -> JSONResponse:
    validate_inbound_secret(request)

    form = await request.form()
    raw_to = str(form.get("to", "")).strip()
    from_email = str(form.get("from", "")).strip()
    subject = str(form.get("subject", "")).strip() or None
    text = str(form.get("text", "")).strip()
    headers_raw = str(form.get("headers", "") or "")

    recipients = [
        addr.strip().lower()
        for addr in re.split(r"[,\s]+", raw_to)
        if addr.strip()
    ]
    client = None
    if recipients:
        client = session.exec(
            select(Client).where(Client.inbound_address.in_(recipients))
        ).first()
    if not client:
        return JSONResponse({"ok": True, "detail": "Client not found for recipient."})

    provider_message_id = parse_message_id(headers_raw) or compute_provider_message_id(
        raw_to, from_email, subject or "", text or ""
    )

    existing = session.exec(
        select(InboundEmail).where(
            InboundEmail.client_id == client.id,
            InboundEmail.provider_message_id == provider_message_id,
        )
    ).first()
    if existing:
        return JSONResponse({"ok": True, "detail": "Already processed"})

    inbound_email = InboundEmail(
        client_id=client.id,
        provider="sendgrid",
        provider_message_id=provider_message_id,
        from_email=from_email,
        to_email=raw_to,
        subject=subject,
        body_text=text,
    )
    session.add(inbound_email)
    session.commit()
    session.refresh(inbound_email)

    enqueue_inbound_processing(inbound_email.id)
    return JSONResponse({"ok": True, "inbound_id": inbound_email.id})


@app.post("/lead/{company_slug}")
async def submit_lead(
    company_slug: str,
    request: Request,
    session: Session = Depends(get_session),
) -> JSONResponse:
    correlation_id = secrets.token_hex(8)
    ip = request.client.host if request.client else "unknown"
    rate_limit(ip)
    log_event(correlation_id, "LEAD_RECEIVED", company_slug=company_slug, ip=ip)

    try:
        data = await request.json()
    except Exception:
        form = await request.form()
        data = dict(form)

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip() or None
    phone = str(data.get("phone", "")).strip() if data.get("phone") is not None else None
    message_text = str(data.get("message", "")).strip() if data.get("message") is not None else ""
    meta = data.get("meta")
    user_agent = request.headers.get("user-agent", "")

    client = session.exec(
        select(Client).where(Client.company_slug == company_slug.lower())
    ).first()
    if not client or not client.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    if not client.inquiry_email:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Client missing inquiry_email",
        )

    meta_payload = {"ip": ip, "user_agent": user_agent}
    if meta is not None:
        meta_payload["meta"] = meta

    lead = Lead(
        client_id=client.id,
        name=name,
        email=email,
        phone=phone,
        message_text=message_text,
        meta_json=json.dumps(meta_payload),
        email_status="pending",
        slack_status="pending",
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    log_event(correlation_id, "LEAD_SAVED", lead_id=lead.id, client_id=client.id)

    from_email = resolve_from_email()
    if not from_email:
        lead.email_status = "failed"
        lead.error_email = "MISSING_FROM_EMAIL"
        session.add(lead)
        session.commit()
        log_event(correlation_id, "EMAIL_SEND_RESULT", status="failed", error="MISSING_FROM_EMAIL")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing verified FROM email",
        )

    api_key = os.getenv("SENDGRID_API_KEY", "").strip()
    if not api_key:
        lead.email_status = "failed"
        lead.error_email = "MISSING_SENDGRID_API_KEY"
        session.add(lead)
        session.commit()
        log_event(correlation_id, "EMAIL_SEND_RESULT", status="failed", error="missing API key")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing SENDGRID_API_KEY",
        )

    sanitized_to = sanitize_header(str(client.inquiry_email))
    subject = sanitize_header(f"New lead from {name}" if name else "New lead")

    body_lines = [
        f"Client: {client.name}",
        f"Lead name: {name or 'N/A'}",
        f"Lead email: {email or 'N/A'}",
    ]
    if phone:
        body_lines.append(f"Phone: {phone}")
    if message_text:
        body_lines.append(f"Message:\n{message_text}")
    if client.business_description:
        body_lines.append(f"\nAbout the business:\n{client.business_description}")
    if client.pricing:
        body_lines.append(f"\nPricing:\n{client.pricing}")
    if meta is not None:
        body_lines.append(f"\nMeta: {json.dumps(meta)}")
    if client.sign_off_name:
        body_lines.append(f"\n{client.sign_off_name}")

    email_payload: dict[str, Any] = {
        "personalizations": [{"to": [{"email": sanitized_to}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": "\n".join(body_lines)}],
    }

    log_event(correlation_id, "EMAIL_SEND_ATTEMPT", to=sanitized_to)
    try:
        with httpx.Client(timeout=10.0) as client_http:
            resp = client_http.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(email_payload),
            )
    except Exception as exc:
        lead.email_status = "failed"
        lead.error_email = str(exc)
        session.add(lead)
        session.commit()
        log_event(correlation_id, "EMAIL_SEND_RESULT", status="failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Email send failed")

    if resp.status_code >= 300:
        lead.email_status = "failed"
        lead.error_email = f"SendGrid {resp.status_code}: {resp.text}"
        session.add(lead)
        session.commit()
        log_event(
            correlation_id,
            "EMAIL_SEND_RESULT",
            status="failed",
            status_code=resp.status_code,
            body=resp.text,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Email rejected by provider",
        )

    lead.email_status = "sent"
    session.add(lead)
    session.commit()
    log_event(
        correlation_id,
        "EMAIL_SEND_RESULT",
        status="sent",
        status_code=resp.status_code,
    )

    slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip() or client.slack_webhook_url
    if slack_webhook:
        slack_message = (
            f"New lead for {client.name}\n"
            f"Name: {name or 'N/A'}\n"
            f"Email: {email or 'N/A'}\n"
            f"Phone: {phone or 'N/A'}\n"
            f"Message: {message_text or 'N/A'}"
        )
        log_event(correlation_id, "SLACK_SEND_ATTEMPT")
        try:
            with httpx.Client(timeout=5.0) as client_http:
                slack_resp = client_http.post(slack_webhook, json={"text": slack_message})
            if slack_resp.status_code >= 300:
                lead.slack_status = "failed"
                lead.error_slack = f"Slack {slack_resp.status_code}: {slack_resp.text}"
                log_event(
                    correlation_id,
                    "SLACK_SEND_RESULT",
                    status="failed",
                    status_code=slack_resp.status_code,
                    body=slack_resp.text,
                )
            else:
                lead.slack_status = "sent"
                log_event(
                    correlation_id,
                    "SLACK_SEND_RESULT",
                    status="sent",
                    status_code=slack_resp.status_code,
                )
        except Exception as exc:
            lead.slack_status = "failed"
            lead.error_slack = str(exc)
            log_event(correlation_id, "SLACK_SEND_RESULT", status="failed", error=str(exc))
    else:
        lead.slack_status = "failed"
        lead.error_slack = "Missing Slack webhook URL"
        log_event(correlation_id, "SLACK_SEND_RESULT", status="failed", error="missing webhook")

    session.add(lead)
    session.commit()

    return JSONResponse({"ok": True, "lead_id": lead.id})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
