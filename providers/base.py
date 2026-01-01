from __future__ import annotations

from typing import Any, Dict, Optional


class EmailProvider:
    def get_authorization_url(self, state: str) -> str:
        raise NotImplementedError

    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        raise NotImplementedError

    def refresh_tokens_if_needed(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def ensure_inbound_subscription(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def list_recent_changes(self, tokens: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def fetch_message(self, tokens: Dict[str, Any], message_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def create_draft(self, tokens: Dict[str, Any], thread_id: Optional[str], to: str, subject: str, body: str) -> Dict[str, Any]:
        raise NotImplementedError

    def send_message(
        self,
        tokens: Dict[str, Any],
        thread_id: Optional[str],
        to: str,
        subject: str,
        body: str,
        reply_to_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError
