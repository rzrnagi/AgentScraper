"""
Unified LLM client for the v1 scraper checkout.
Supports Anthropic, OpenAI, and Ollama using environment variables from setup.py.
"""
import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2"


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @classmethod
    def from_env(cls) -> "LLMClient":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if provider == "anthropic":
            return cls(provider=provider, model=ANTHROPIC_MODEL)
        if provider == "openai":
            return cls(provider=provider, model=OPENAI_MODEL)
        if provider == "ollama":
            return cls(provider=provider, model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
        raise ValueError("LLM_PROVIDER must be one of: anthropic, openai, ollama")

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            return await self._complete_anthropic(prompt, max_tokens)
        if self.provider == "openai":
            return await self._complete_openai(prompt, max_tokens)
        if self.provider == "ollama":
            return await self._complete_ollama(prompt, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def complete_json(self, prompt: str, max_tokens: int = 1024) -> dict:
        text = await self.complete(prompt, max_tokens)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
            raise

    async def _complete_anthropic(self, prompt: str, max_tokens: int) -> str:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Run python3 setup.py")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, max_tokens: int) -> str:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Run python3 setup.py")

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, max_tokens: int) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
