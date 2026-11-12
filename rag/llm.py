"""The "G" in RAG: an LLM behind a three-method interface.

Deliberately no SDK dependencies - every provider is one HTTP POST you can
read. Pick one with LLM_PROVIDER in .env:

    ollama      free, local, private.  `ollama pull llama3.2` then `ollama serve`
    openai      OPENAI_API_KEY
    gemini      GEMINI_API_KEY
    anthropic   ANTHROPIC_API_KEY
    extractive  no LLM at all (handled in generate.py, not here)

If a provider fails at runtime, the pipeline falls back to extractive mode and
says so, rather than showing you a stack trace.
"""
from __future__ import annotations

import json
import os

import config


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def _post(self, url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
        import requests

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:
            hint = {
                "ollama": " - is `ollama serve` running?",
                "openai": " - check your network and OPENAI_API_KEY",
                "gemini": " - check your network and GEMINI_API_KEY",
                "anthropic": " - check your network and ANTHROPIC_API_KEY",
            }.get(self.name, "")
            raise LLMError(
                f"{self.name} is not reachable ({type(exc).__name__}){hint}"
            ) from exc
        if response.status_code >= 400:
            raise LLMError(f"{self.name} returned {response.status_code}: {response.text[:300]}")
        return response.json()


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str = "") -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = config.OLLAMA_HOST.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            f"{self.host}/api/chat",
            {},
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": config.TEMPERATURE},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return (data.get("message") or {}).get("content", "").strip()


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str = "") -> None:
        self.key = os.getenv("OPENAI_API_KEY", "")
        if not self.key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.key}"},
            {
                "model": self.model,
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_ANSWER_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return data["choices"][0]["message"]["content"].strip()


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, model: str = "") -> None:
        self.key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.key}",
            {},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": config.TEMPERATURE,
                    "maxOutputTokens": config.MAX_ANSWER_TOKENS,
                },
            },
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("gemini returned no candidates (often a safety block)")
        return "".join(
            part.get("text", "")
            for part in candidates[0]["content"]["parts"]
        ).strip()


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model: str = "") -> None:
        self.key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "system": system,
                "max_tokens": config.MAX_ANSWER_TOKENS,
                "temperature": config.TEMPERATURE,
                "messages": [{"role": "user", "content": user}],
            },
        )
        return "".join(
            block.get("text", "") for block in data.get("content", [])
        ).strip()


def get_llm(provider: str | None = None) -> BaseLLM | None:
    """Return None for extractive mode - generate.py handles that path."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider in {"extractive", "none", ""}:
        return None
    builders = {
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
        "gemini": GeminiLLM,
        "anthropic": AnthropicLLM,
    }
    if provider not in builders:
        raise SystemExit(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Choose from: extractive, {', '.join(builders)}"
        )
    return builders[provider](config.LLM_MODEL)


def describe(llm: BaseLLM | None) -> str:
    if llm is None:
        return "extractive (no LLM)"
    model = getattr(llm, "model", "?")
    return f"{llm.name}:{model}"


def dumps(obj) -> str:  # tiny helper used by the CLI's --json flag
    return json.dumps(obj, indent=2, ensure_ascii=False)
