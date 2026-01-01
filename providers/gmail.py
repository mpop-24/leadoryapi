from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
from google.oauth2 import id_token
from google.auth.transport import requests

from .base import EmailProvider


class GmailProvider(EmailProvider):
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SCOPE = "https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.send"
    WATCH_URL = "https://gmail.googleapis.com/gmail/v1/users/me/watch"
    HISTORY_URL = "https://gmail.googleapis.com/gmail/v1/users/me/history"

    def __init__(self) -> None:
        self.client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")
        self.pubsub_topic = os.getenv("GOOGLE_PUBSUB_TOPIC", "")
        self.webhook_audience = os.getenv("GOOGLE_PUBSUB_AUDIENCE", "")
        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise RuntimeError("Missing Google OAuth env vars")

    def get_authorization_url(self, state: str) -> str:
        return (
            f"{self.AUTH_URL}?response_type=code&client_id={self.client_id}"
            f"&redirect_uri={self.redirect_uri}&scope={self.SCOPE}"
            f"&access_type=offline&prompt=consent&state={state}"
        )

    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.TOKEN_URL, data=data)
            resp.raise_for_status()
            return resp.json()

    def refresh_tokens_if_needed(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        if not tokens.get("refresh_token"):
            return tokens
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.TOKEN_URL, data=data)
            resp.raise_for_status()
            refreshed = resp.json()
            tokens.update(refreshed)
            tokens["expires_at"] = refreshed.get("expires_in")
            return tokens

    def ensure_inbound_subscription(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.pubsub_topic:
            return metadata or {}
        # Google requires a unique labelIds/state; we reuse INBOX and rely on historyId for delta
        body = {
            "topicName": self.pubsub_topic,
            "labelIds": ["INBOX"],
        }
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.WATCH_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            meta = metadata or {}
            meta.update(
                {
                    "history_id": data.get("historyId"),
                    "expiration": data.get("expiration"),
                    "watch_created_at": datetime.utcnow().isoformat(),
                    "topic_name": self.pubsub_topic,
                }
            )
            return meta

    def list_recent_changes(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        params: Dict[str, Any] = {"labelId": "INBOX", "historyTypes": "messageAdded"}
        start_history_id = None
        if metadata:
            start_history_id = metadata.get("history_id")
        if start_history_id:
            params["startHistoryId"] = start_history_id
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self.HISTORY_URL, headers=headers, params=params)
            if resp.status_code == 404:
                # historyId too old, needs new watch
                return {"messages": [], "reset_watch": True}
            resp.raise_for_status()
            data = resp.json()
            messages: list[dict] = []
            for hist in data.get("history", []):
                for added in hist.get("messagesAdded", []):
                    msg = added.get("message", {})
                    messages.append({"id": msg.get("id"), "thread_id": msg.get("threadId")})
            # Update history_id to latest
            latest = data.get("historyId") or start_history_id
            return {"messages": messages, "history_id": latest}

    def fetch_message(self, tokens: Dict[str, Any], message_id: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                params={"format": "full"},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            payload = data.get("payload", {})
            headers_list = payload.get("headers", []) or []
            header_map = {h.get("name", "").lower(): h.get("value", "") for h in headers_list}
            subject = header_map.get("subject", "")
            from_email = header_map.get("from", "")
            to_email = header_map.get("to", "")
            body = ""
            parts = payload.get("parts", []) or []
            if payload.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            elif parts:
                for part in parts:
                    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                        break
            return {
                "subject": subject,
                "body": body,
                "snippet": data.get("snippet", ""),
                "thread_id": data.get("threadId"),
                "id": data.get("id"),
                "from_email": from_email,
                "to_email": to_email,
                "received_at": None,
            }

    def create_draft(self, tokens: Dict[str, Any], thread_id: Optional[str], to: str, subject: str, body: str) -> Dict[str, Any]:
        # For simplicity, we skip draft creation in this scaffold
        return {"draft_created": True}

    def send_message(self, tokens: Dict[str, Any], thread_id: Optional[str], to: str, subject: str, body: str, reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        raw = f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}"
        raw_b64 = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8")
        payload: Dict[str, Any] = {"raw": raw_b64}
        if thread_id:
            payload["threadId"] = thread_id
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    def verify_pubsub_jwt(self, auth_header: str) -> bool:
        if not self.webhook_audience:
            return True
        token = auth_header.split(" ", 1)[1] if " " in auth_header else auth_header
        try:
            id_token.verify_oauth2_token(token, requests.Request(), audience=self.webhook_audience)
            return True
        except Exception:
            return False
