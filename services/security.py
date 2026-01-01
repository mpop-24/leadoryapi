from __future__ import annotations

import hmac
import os
import time
from hashlib import sha256
from typing import Optional, Tuple

SECRET = os.getenv("JWT_SECRET", "").encode()
if not SECRET:
    raise RuntimeError("JWT_SECRET missing for state signing")


def sign_state(client_id: int, provider: str, ttl_seconds: int = 600) -> str:
    ts = int(time.time())
    payload = f"{client_id}:{provider}:{ts}"
    sig = hmac.new(SECRET, payload.encode(), sha256).hexdigest()
    return f"{payload}:{sig}:{ttl_seconds}"


def verify_state(state: str) -> Optional[Tuple[int, str]]:
    try:
        client_id_str, provider, ts_str, sig, ttl_str = state.split(":", 4)
        ts = int(ts_str)
        ttl = int(ttl_str)
        expected = hmac.new(SECRET, f"{client_id_str}:{provider}:{ts}".encode(), sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        if time.time() > ts + ttl:
            return None
        return int(client_id_str), provider
    except Exception:
        return None
