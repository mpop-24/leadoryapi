from __future__ import annotations

import base64
import os
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_ENC_KEY_RAW = os.getenv("ENCRYPTION_KEY", "").strip()
if not _ENC_KEY_RAW:
    raise RuntimeError("ENCRYPTION_KEY is required for token encryption")

try:
    if len(_ENC_KEY_RAW) == 64:  # hex 32 bytes
        _KEY = bytes.fromhex(_ENC_KEY_RAW)
    else:
        _KEY = base64.b64decode(_ENC_KEY_RAW)
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Invalid ENCRYPTION_KEY, must be 32-byte hex or base64") from exc

if len(_KEY) != 32:
    raise RuntimeError("ENCRYPTION_KEY must be 32 bytes")


def encrypt(plaintext: str) -> str:
    aesgcm = AESGCM(_KEY)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("utf-8")


def decrypt(ciphertext: str) -> str:
    data = base64.b64decode(ciphertext)
    nonce, ct = data[:12], data[12:]
    aesgcm = AESGCM(_KEY)
    pt = aesgcm.decrypt(nonce, ct, None)
    return pt.decode("utf-8")
