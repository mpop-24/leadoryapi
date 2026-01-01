from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import engine
from models import SQLModel
from routers import admin, health, oauth, webhooks, leads

app = FastAPI(title="Connected Inbox Lead Assistant", version="1.0.0")

origins_env = os.getenv("CORS_ORIGINS", "").strip()
origins: List[str] = [o.strip() for o in origins_env.split(",") if o.strip()] or [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)
app.include_router(oauth.router)
app.include_router(webhooks.router)
app.include_router(leads.router)
app.include_router(health.router)


@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)


@app.get("/")
async def root():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
