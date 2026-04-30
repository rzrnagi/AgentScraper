"""
Unified LLM client — wraps Anthropic, OpenAI, and Ollama under one interface.

Usage:
    client = LLMClient.from_env()
    response = await client.complete("Your prompt here", max_tokens=512)
"""
import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_MODELS = {
    "fast":  "claude-haiku-4-5-20251001",
    "smart": "claude-sonnet-4-6",
}
OPENAI_MODELS = {
    "fast":  "gpt-4o-mini",
    "smart": "gpt-4o",
}
DEFAULT_OLLAMA_MODEL = "llama3.2"


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @classmethod
    def from_env(cls, tier: str = "fast") -> "LLMClient":
        """
        Build client from environment variables set by setup.py.
        tier: "fast" (cheap/quick) or "smart" (more capable, costs more)
        """
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

        if provider == "anthropic":
            model = ANTHROPIC_MODELS.get(tier, ANTHROPIC_MODELS["fast"])
        elif provider == "openai":
            model = OPENAI_MODELS.get(tier, OPENAI_MODELS["fast"])
        elif provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

        logger.info("LLM client: provider=%s model=%s", provider, model)
        return cls(provider=provider, model=model)

    async def complete(self, prompt: str, max_tokens: int = 1024, system: str = None) -> str:
        """Send a prompt, return the text response."""
        if self.provider == "anthropic":
            return await self._complete_anthropic(prompt, max_tokens, system)
        elif self.provider == "openai":
            return await self._complete_openai(prompt, max_tokens, system)
        elif self.provider == "ollama":
            return await self._complete_ollama(prompt, max_tokens, system)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def complete_json(self, prompt: str, max_tokens: int = 1024, system: str = None) -> dict:
        """Send a prompt, parse and return JSON response."""
        text = await self.complete(prompt, max_tokens, system)
        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Fix lone backslashes that aren't valid JSON escape sequences
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Extract outermost JSON object and retry
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                chunk = match.group()
                chunk = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', chunk)
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    pass
            raise

    async def _complete_anthropic(self, prompt: str, max_tokens: int, system: str) -> str:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Run python3 setup.py")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        return response.content[0].text

    async def _complete_openai(self, prompt: str, max_tokens: int, system: str) -> str:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Run python3 setup.py")
        client = openai.AsyncOpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content

    async def _complete_ollama(self, prompt: str, max_tokens: int, system: str) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]
