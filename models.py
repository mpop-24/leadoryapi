from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint


class ReplyMode(str, enum.Enum):
    auto_send = "auto_send"
    draft = "draft"


class Provider(str, enum.Enum):
    gmail = "gmail"
    microsoft = "microsoft"


class ConnectionStatus(str, enum.Enum):
    active = "active"
    needs_reconnect = "needs_reconnect"
    revoked = "revoked"
    error = "error"


class Direction(str, enum.Enum):
    inbound = "inbound"
    outbound = "outbound"


class LeadSource(str, enum.Enum):
    email_inbox = "email_inbox"
    form = "form"


class StatusGeneric(str, enum.Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"


class AIStatus(str, enum.Enum):
    pending = "pending"
    drafted = "drafted"
    failed = "failed"


class JobStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})


class Client(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True)
    business_name: str
    inquiry_email: Optional[EmailStr] = Field(default=None, index=True)
    pricing: Optional[str] = None
    business_description: Optional[str] = None
    sign_off_name: Optional[str] = None
    mimic_email: Optional[EmailStr] = None
    slack_webhook_url: Optional[str] = None
    reply_mode: ReplyMode = Field(default=ReplyMode.auto_send)

    connections: list["MailboxConnection"] = Relationship("MailboxConnection", back_populates="client")
    threads: list["EmailThread"] = Relationship("EmailThread", back_populates="client")
    leads: list["Lead"] = Relationship("Lead", back_populates="client")


class MailboxConnection(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    provider: Provider
    connected_email: EmailStr
    access_token_encrypted: str
    refresh_token_encrypted: Optional[str] = None
    expires_at: Optional[datetime] = None
    status: ConnectionStatus = Field(default=ConnectionStatus.active, index=True)
    last_sync_at: Optional[datetime] = None
    provider_metadata: Optional[str] = None

    client: Client = Relationship("Client", back_populates="connections")
    threads: list["EmailThread"] = Relationship("EmailThread", back_populates="connection")


class EmailThread(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    connection_id: Optional[int] = Field(foreign_key="mailboxconnection.id", index=True)
    provider: Provider
    provider_thread_id: str = Field(index=True)
    subject: Optional[str] = None
    last_message_at: Optional[datetime] = None
    __table_args__ = (UniqueConstraint("client_id", "provider_thread_id", name="uq_client_thread"),)

    client: Client = Relationship("Client", back_populates="threads")
    connection: Optional[MailboxConnection] = Relationship("MailboxConnection", back_populates="threads")
    messages: list["EmailMessage"] = Relationship("EmailMessage", back_populates="thread")
    leads: list["Lead"] = Relationship("Lead", back_populates="thread")


class EmailMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    thread_id: Optional[int] = Field(foreign_key="emailthread.id", index=True)
    provider: Provider
    provider_message_id: str = Field(index=True)
    direction: Direction
    from_email: str
    to_email: str
    cc_json: Optional[str] = None
    subject: Optional[str] = None
    body_text: Optional[str] = None
    snippet: Optional[str] = None
    received_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    __table_args__ = (UniqueConstraint("client_id", "provider_message_id", name="uq_client_message"),)

    thread: Optional[EmailThread] = Relationship("EmailThread", back_populates="messages")


class Lead(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id", index=True)
    thread_id: Optional[int] = Field(foreign_key="emailthread.id", index=True)
    lead_name: Optional[str] = None
    lead_email: Optional[EmailStr] = Field(default=None, index=True)
    lead_phone: Optional[str] = None
    message_text: Optional[str] = None
    source: LeadSource
    email_status: StatusGeneric = Field(default=StatusGeneric.pending)
    slack_status: StatusGeneric = Field(default=StatusGeneric.pending)
    ai_status: AIStatus = Field(default=AIStatus.pending)
    error_email: Optional[str] = None
    error_slack: Optional[str] = None
    error_ai: Optional[str] = None
    correlation_id: Optional[str] = Field(default=None, index=True)

    client: Client = Relationship("Client", back_populates="leads")
    thread: Optional[EmailThread] = Relationship("EmailThread", back_populates="leads")
    drafts: list["AIDraft"] = Relationship("AIDraft", back_populates="lead")


class AIDraft(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lead_id: int = Field(foreign_key="lead.id", index=True)
    prompt_text: str
    draft_text: str
    model_name: str
    sent_at: Optional[datetime] = None

    lead: Lead = Relationship("Lead", back_populates="drafts")


class Job(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_type: str
    payload: str
    status: JobStatus = Field(default=JobStatus.pending, index=True)
    attempts: int = Field(default=0)
    last_error: Optional[str] = None
    scheduled_at: Optional[datetime] = None
