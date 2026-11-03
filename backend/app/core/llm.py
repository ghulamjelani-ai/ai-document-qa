"""Generate a chat answer. Two interchangeable providers: Ollama and OpenAI.

Both are called over HTTP so no heavy SDK is required. The OpenAI client also
works with any OpenAI-compatible endpoint (base_url is configurable).
"""
from __future__ import annotations

import httpx

from app.config import Settings


class LLMError(RuntimeError):
    pass


class LLM:
    name: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, settings: Settings):
        self._url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        self.name = settings.ollama_chat_model
        self._timeout = settings.request_timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama chat request failed ({self.name}): {exc}") from exc


class OpenAILLM(LLM):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set but llm_provider=openai")
        self._url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        self._key = settings.openai_api_key
        self.name = settings.openai_chat_model
        self._timeout = settings.request_timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI chat request failed ({self.name}): {exc}") from exc


def build_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaLLM(settings)
    if provider == "openai":
        return OpenAILLM(settings)
    raise LLMError(f"Unknown llm_provider: {settings.llm_provider}")
