from __future__ import annotations

import os
import time
from typing import Optional

import httpx


def send_slack(webhook_url: str, message: str, retries: int = 3) -> tuple[bool, Optional[str]]:
    delay = 1
    for _ in range(retries):
        try:
            resp = httpx.post(webhook_url, json={"text": message}, timeout=5.0)
            if resp.status_code < 300:
                return True, None
            err = f"Slack {resp.status_code}: {resp.text}"
        except Exception as exc:  # pragma: no cover - best effort
            err = str(exc)
        time.sleep(delay)
        delay *= 2
    return False, err


def get_client_webhook(client_webhook: Optional[str]) -> Optional[str]:
    return client_webhook or os.getenv("SLACK_WEBHOOK_URL", "").strip() or None
