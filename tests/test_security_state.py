from __future__ import annotations

import os
import time
import importlib
import pytest


def setup_module(module):
    os.environ.setdefault("JWT_SECRET", "testsecret1234567890")
    import services.security as sec
    importlib.reload(sec)


def test_state_sign_verify_round_trip():
    from services.security import sign_state, verify_state

    state = sign_state(123, "gmail", ttl_seconds=5)
    parsed = verify_state(state)
    assert parsed == (123, "gmail")


def test_state_expired():
    from services.security import sign_state, verify_state

    state = sign_state(123, "gmail", ttl_seconds=0)
    time.sleep(1)
    assert verify_state(state) is None


def test_state_tamper():
    from services.security import sign_state, verify_state

    state = sign_state(123, "gmail", ttl_seconds=5)
    bad = state + "x"
    assert verify_state(bad) is None
