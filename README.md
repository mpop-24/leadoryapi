Connected Inbox Lead Assistant
==============================

FastAPI + SQLModel + Postgres service that connects to real mailboxes (Gmail/Outlook) via OAuth, ingests inbound leads, drafts AI replies, auto-sends from the connected inbox, and notifies Slack. Webhooks with polling fallback, encrypted tokens, and a DB-backed job queue/worker.

Requirements
------------
- Python 3.12+
- Postgres database
- Google/Microsoft OAuth apps configured (see below)
- Optional: OpenAI API key (for AI drafts), Slack webhook

Environment Variables
---------------------
- Core: `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `JWT_SECRET`, `DATABASE_URL`, `ENCRYPTION_KEY`, `CORS_ORIGINS`, `BASE_URL`
- Gmail: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `GOOGLE_PUBSUB_TOPIC`, `GOOGLE_PUBSUB_AUDIENCE`, `GMAIL_WEBHOOK_SECRET`
- Microsoft: `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_OAUTH_REDIRECT_URI`, `MICROSOFT_TENANT`, `MICROSOFT_SUBSCRIPTION_CLIENT_STATE_SECRET`, `MICROSOFT_NOTIFICATION_URL`
- Optional: `OPENAI_API_KEY`, `SLACK_WEBHOOK_URL`

OAuth Setup
-----------
- Gmail: Create OAuth client (web), add redirect URI, enable Gmail API. Create Pub/Sub topic + push subscription to `/webhooks/gmail/push` with audience/secret.
- Microsoft: Register app, scopes Mail.ReadWrite/Mail.Send, add redirect URI. Configure notification URL `/webhooks/microsoft/notifications`; set clientState secret.

Running Locally
---------------
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ADMIN_USERNAME=admin ADMIN_PASSWORD=pass JWT_SECRET=devjwt
export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/leadory
export ENCRYPTION_KEY=<32-byte-hex-or-base64>
# set provider envs as above
alembic upgrade head
uvicorn main:app --host 0.0.0.0 --port 8000
# in separate shells:
python -m worker.worker
python -m worker.scheduler
```

Processes
---------
- Web API: `uvicorn main:app`
- Worker: `python -m worker.worker` (processes job table)
- Scheduler: `python -m worker.scheduler` (enqueues poll/renew jobs)

Endpoints (admin routes require bearer token)
---------------------------------------------
- Auth: `POST /admin/login`
- Clients: `GET/POST/PATCH /admin/clients`, `GET /admin/clients/{id}`, `GET /admin/clients/{id}/connection-status`, `POST /admin/clients/{id}/disconnect`
- Connections: `POST /admin/clients/{id}/connect/google`, `POST /admin/clients/{id}/connect/microsoft`
- Leads: `GET /admin/leads`, `GET /admin/leads/{lead_id}`, public `POST /lead/{company_slug}`
- Webhooks: `POST /webhooks/gmail/push` (Gmail Pub/Sub), `POST /webhooks/microsoft/notifications` (Graph)
- Health: `/health`, `/health/db`, `/health/providers`

Notes
-----
- Tokens are encrypted (AES-GCM) via `ENCRYPTION_KEY`; HMAC-signed state used for OAuth.
- Webhooks require secrets (Gmail: Pub/Sub audience + `GMAIL_WEBHOOK_SECRET`; Microsoft: `MICROSOFT_SUBSCRIPTION_CLIENT_STATE_SECRET`).
- Job queue is DB-backed; polling every 5m; renewals every 30m with backoff on failures.
- Scopes: Gmail `gmail.modify` + `gmail.send`; Microsoft `Mail.ReadWrite` + `Mail.Send`.
- AI drafts: uses `OPENAI_API_KEY` if set; spam/unsafe guard prevents auto-send on obvious spam.

Testing
-------
- Unit: `pytest -q` (integration with live providers not covered here).
- Manual: connect a mailbox, send an email, verify lead stored, Slack (if configured), AI draft, and auto-reply from the connected account.***
