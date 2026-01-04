from __future__ import annotations

import base64
import json
import os
import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

# Ensure JWT_SECRET is set for imports
os.environ.setdefault("JWT_SECRET", "testsecret1234567890")
os.environ.setdefault("GMAIL_WEBHOOK_SECRET", "test-secret")

from main import app


def test_gmail_webhook_no_auth():
    with TestClient(app) as client:
        payload = {
            "message": {
                "data": base64.b64encode(json.dumps({"emailAddress": "test@example.com", "historyId": "1"}).encode()).decode()
            }
        }
        resp = client.post("/webhooks/gmail/push", json=payload)
        assert resp.status_code == 401  # secret required


def test_ms_validation_token():
    with TestClient(app) as client:
        resp = client.post("/webhooks/microsoft/notifications?validationToken=abc", json={})
        assert resp.status_code == 200
        assert resp.text == '"abc"'


def test_ms_notifications_unauthorized():
    with TestClient(app) as client:
        resp = client.post(
            "/webhooks/microsoft/notifications",
            json={"value": [{"subscriptionId": "sub1", "clientState": "bad", "resourceData": {"id": "msg"}}]},
        )
        assert resp.status_code == 401
