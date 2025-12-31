Leadory Email Engine
====================

Minimal FastAPI + Postgres service for ingesting inbound emails per client, classifying, and replying via SendGrid. All client configuration lives in the database.

Requirements
------------
- Python 3.12+
- Postgres database
- SendGrid account (Inbound Parse + Mail Send)
- OpenAI API key (for classification + reply generation)

Environment Variables
---------------------
- `DATABASE_URL` — Postgres connection string (e.g. `postgresql+psycopg://user:pass@host:5432/db`)
- `SENDGRID_API_KEY` — SendGrid API key with Mail Send + Inbound Parse permissions
- `INBOUND_WEBHOOK_SECRET` — shared secret required by `/webhooks/sendgrid/inbound`
- `INBOUND_EMAIL_DOMAIN` — domain for inbound addresses (default `inbound.leadory.co`)
- `OPENAI_KEY` — OpenAI API key
- `ADMIN_USERNAME`, `ADMIN_PASSWORD` — dashboard/admin login
- `JWT_SECRET` — secret for issuing admin tokens
- Optional: `ADMIN_KEY` (header override), `JWT_EXPIRES_SECONDS`

Database Models
---------------
- **Client**: name, `company_slug`, `is_active`, `inbound_address`, `from_email`, optional `reply_to_email`, `cc_email`, `slack_webhook_url`, timestamps.
- **InboundEmail**: `client_id`, provider (`sendgrid`), `provider_message_id`, `from_email`, `to_email`, subject, body, `received_at`, classification (`YES/NO/ERROR`), reply_status (`SENT/SKIPPED/FAILED`), timestamps, optional `error`. Unique `(client_id, provider_message_id)`.

SendGrid Inbound Parse Setup
----------------------------
1. Configure domain/MX: point `inbound.leadory.co` MX to SendGrid Inbound Parse per SendGrid docs.
2. Inbound Parse settings:
   - Hostname: `inbound.leadory.co`
   - URL: `https://<your-domain>/webhooks/sendgrid/inbound?secret=INBOUND_WEBHOOK_SECRET`
   - Spam check disabled; POST format `multipart/form-data`.
3. Each client uses an address like `client_<id>@inbound.leadory.co`. Leads should forward/route to that address.

SendGrid Outbound Sending
-------------------------
- Verify a Sender Identity or domain authenticate the sending domain for `from_email`.
- Mail send uses SendGrid v3 API with headers `X-Leadory-Client-ID` and `X-Leadory-Inbound-ID`.
- If sending from the Leadory domain, configure `reply_to_email` so customer replies go to the client owner.

Running Locally
---------------
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/leadory
export SENDGRID_API_KEY=...
export INBOUND_WEBHOOK_SECRET=dev-secret
export OPENAI_KEY=...
export ADMIN_USERNAME=admin ADMIN_PASSWORD=pass JWT_SECRET=jwtsecret
uvicorn main:app --reload --port 8000
```

Sample Inbound Webhook Test
---------------------------
```bash
curl -X POST "http://localhost:8000/webhooks/sendgrid/inbound?secret=dev-secret" \
  -F "from=lead@example.com" \
  -F "to=client_1@inbound.leadory.co" \
  -F "subject=Test Lead" \
  -F "text=Hello, I want to buy your product" \
  -F "headers=Message-ID: <test123@example.com>"
```
Response returns quickly; processing happens in background.

Admin & Dashboard Endpoints
---------------------------
- `POST /admin/login` — get bearer token
- `GET /admin/clients` — list clients
- `POST /admin/clients` — create client (auto-generates inbound address if omitted)
- `PUT /admin/clients/{id}` — update client fields
- `POST /admin/clients/{id}/toggle` — pause/resume
- `GET /admin/inbound-emails` — recent inbound emails (optional `client_id`)

Embed/Form Routing
------------------
- Use `company_slug` to build embed URLs or links (e.g. `/lead/{company_slug}`).
- Form submission should POST to `/lead/{company_slug}` with `name`, `email`, optional `phone`, `message`, `meta`. It creates an `InboundEmail` row and triggers the same reply pipeline.

Notes
-----
- Inbound webhook validates `INBOUND_WEBHOOK_SECRET` and immediately returns 200 after storing the email.
- Replies are idempotent via `(client_id, provider_message_id)` uniqueness.
- Slack notifications fire when configured; failures trigger an alert.***
