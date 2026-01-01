from __future__ import annotations

import os
import re
from typing import Optional

import httpx


DEFAULT_MODEL = "gpt-4o-mini"

def is_spam_or_unsafe(text: str) -> bool:
    lowered = text.lower()
    spam_signals = ["viagra", "lottery", "crypto", "wire money", "inheritance", "sex", "kill", "suicide", "threat"]
    short = len(lowered.strip()) < 10
    return short or any(sig in lowered for sig in spam_signals)


def build_prompt(text: str, business_name: str, business_description: Optional[str], pricing: Optional[str], sign_off_name: Optional[str], mimic_email: Optional[str]) -> str:
    guidance_parts = []
    if business_description:
        guidance_parts.append(f"Business description: {business_description}")
    if pricing:
        guidance_parts.append(f"Pricing guidance: {pricing}")
    if mimic_email:
        guidance_parts.append(f"Write in a similar tone to this style: {mimic_email}")
    if sign_off_name:
        guidance_parts.append(f"Preferred sign-off: {sign_off_name}")
    guidance = "\n".join(guidance_parts)
    prompt = (
        "You are composing a concise, friendly response to a new inbound lead. "
        f"Client: {business_name}. Use the guidance below as context; do not copy verbatim unless it directly answers the question. "
        "Write 3-6 sentences, ask one clarifying question, and keep it professional.\n\n"
        f"Guidance:\n{guidance}\n\nInbound message:\n{text}\n"
    )
    return prompt


def generate_draft(text: str, business_name: str, business_description: Optional[str], pricing: Optional[str], sign_off_name: Optional[str], mimic_email: Optional[str], model: str = DEFAULT_MODEL) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    prompt = build_prompt(text, business_name, business_description, pricing, sign_off_name, mimic_email)
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return None
