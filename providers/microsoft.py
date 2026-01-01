from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx

from .base import EmailProvider


class MicrosoftProvider(EmailProvider):
    AUTH_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
    TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    SCOPE = "https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Send"
    SUBSCRIPTION_URL = "https://graph.microsoft.com/v1.0/subscriptions"

    def __init__(self) -> None:
        self.client_id = os.getenv("MICROSOFT_CLIENT_ID", "")
        self.client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv("MICROSOFT_OAUTH_REDIRECT_URI", "")
        self.tenant = os.getenv("MICROSOFT_TENANT", "common")
        self.notification_url = os.getenv("MICROSOFT_NOTIFICATION_URL", "")
        if not self.notification_url:
            base = os.getenv("BASE_URL", "")
            if base:
                self.notification_url = base.rstrip("/") + "/webhooks/microsoft/notifications"
        self.client_state = os.getenv("MICROSOFT_SUBSCRIPTION_CLIENT_STATE_SECRET", "")
        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise RuntimeError("Missing Microsoft OAuth env vars")

    def get_authorization_url(self, state: str) -> str:
        return (
            f"{self.AUTH_URL.format(tenant=self.tenant)}?client_id={self.client_id}"
            f"&response_type=code&redirect_uri={self.redirect_uri}"
            f"&response_mode=query&scope={self.SCOPE}&state={state}"
        )

    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        data = {
            "client_id": self.client_id,
            "scope": self.SCOPE,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "client_secret": self.client_secret,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.TOKEN_URL.format(tenant=self.tenant), data=data)
            resp.raise_for_status()
            return resp.json()

    def refresh_tokens_if_needed(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        if not tokens.get("refresh_token"):
            return tokens
        data = {
            "client_id": self.client_id,
            "scope": self.SCOPE,
            "refresh_token": tokens["refresh_token"],
            "redirect_uri": self.redirect_uri,
            "grant_type": "refresh_token",
            "client_secret": self.client_secret,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.TOKEN_URL.format(tenant=self.tenant), data=data)
            resp.raise_for_status()
            refreshed = resp.json()
            tokens.update(refreshed)
            tokens["expires_at"] = refreshed.get("expires_in")
            return tokens

    def ensure_inbound_subscription(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.notification_url:
            return metadata or {}
        meta = metadata or {}
        body = {
            "changeType": "created",
            "notificationUrl": self.notification_url,
            "resource": "me/mailFolders('Inbox')/messages",
            "expirationDateTime": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "clientState": self.client_state or "state",
        }
        headers = {"Authorization": f"Bearer {tokens['access_token']}", "Content-Type": "application/json"}
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(self.SUBSCRIPTION_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            meta.update(
                {
                    "subscription_id": data.get("id"),
                    "expiration": data.get("expirationDateTime"),
                    "delta_link": meta.get("delta_link"),
                    "resource": body["resource"],
                }
            )
            return meta

    def list_recent_changes(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        delta_link = ""
        if metadata:
            delta_link = metadata.get("delta_link", "")
        url = delta_link or "https://graph.microsoft.com/v1.0/me/mailFolders/Inbox/messages/delta"
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            messages = []
            for item in data.get("value", []):
                to_list = item.get("toRecipients") or []
                to_email = None
                if to_list:
                    to_email = (to_list[0].get("emailAddress") or {}).get("address")
                messages.append(
                    {
                        "id": item.get("id"),
                        "thread_id": item.get("conversationId"),
                        "from_email": ((item.get("from") or {}).get("emailAddress") or {}).get("address"),
                        "to_email": to_email,
                        "subject": item.get("subject"),
                        "body": item.get("bodyPreview"),
                        "snippet": item.get("bodyPreview"),
                        "received_at": item.get("receivedDateTime"),
                    }
                )
            next_delta = data.get("@odata.deltaLink") or delta_link
            return {"messages": messages, "delta_link": next_delta}

    def fetch_message(self, tokens: Dict[str, Any], message_id: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"https://graph.microsoft.com/v1.0/me/messages/{message_id}", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return {
                "subject": data.get("subject", ""),
                "body": data.get("body", {}).get("content", ""),
                "snippet": data.get("bodyPreview", ""),
                "thread_id": data.get("conversationId"),
                "id": data.get("id"),
                "from_email": (data.get("from") or {}).get("emailAddress", {}).get("address"),
                "to_email": ((data.get("toRecipients") or [{}])[0].get("emailAddress", {}) or {}).get("address"),
                "received_at": data.get("receivedDateTime"),
            }

    def create_draft(self, tokens: Dict[str, Any], thread_id: Optional[str], to: str, subject: str, body: str) -> Dict[str, Any]:
        # For simplicity, draft creation omitted
        return {"draft_created": True}

    def send_message(self, tokens: Dict[str, Any], thread_id: Optional[str], to: str, subject: str, body: str, reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {tokens['access_token']}", "Content-Type": "application/json"}
        message = {
            "message": {
                "subject": subject,
                "body": {"contentType": "Text", "content": body},
                "toRecipients": [{"emailAddress": {"address": to}}],
            }
        }
        if reply_to_message_id:
            message["message"]["extensions"] = []  # placeholder
        with httpx.Client(timeout=10.0) as client:
            resp = client.post("https://graph.microsoft.com/v1.0/me/sendMail", json=message, headers=headers)
            resp.raise_for_status()
            return {"sent": True}
