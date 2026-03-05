from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str


class LLMClient:
    """
    Hybrid LLM client.

    - If OPENAI_API_KEY is available and the OpenAI package is installed,
      it can call OpenAI.
    - Otherwise, it falls back to a manual mode (user-provided input).
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, model: str = "gpt-4.1-mini") -> LLMResponse:
        """
        Generate text from an LLM. If no API key is present, raise a RuntimeError
        so the caller can fall back to manual mode.
        """
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use manual mode fallback.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK not installed. Install 'openai' or use manual mode."
            ) from exc

        client = OpenAI(api_key=self.api_key)

        # Minimal, robust call
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful risk analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, provider="openai")
