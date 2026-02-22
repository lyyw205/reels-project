"""CreativeLLM — shared LLM wrapper for creative team agents."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypeVar, Type

from pydantic import BaseModel, ValidationError

from reels.production.cache import ResponseCache

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMParsingError(Exception):
    """Raised when LLM response cannot be parsed after retries."""


class CreativeLLM:
    """Shared LLM wrapper for creative team agents.

    Reuses patterns from FeatureExtractor (semaphore + cache + retry)
    and ClaudeVisionBackend (JSON extraction from markdown).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("creative_team", {})
        self._cache = ResponseCache(cfg.get("cache_dir", ".cache/creative"))
        self._semaphore = asyncio.Semaphore(cfg.get("max_concurrent", 3))
        self._client: Any = None  # Lazy init (anthropic.AsyncAnthropic)
        self._default_model = cfg.get("default_model", "claude-sonnet-4-20250514")

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        response_model: Type[T],
        agent_role: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> T:
        """Generate structured output from LLM with Pydantic validation.

        Args:
            system: System prompt for the agent.
            prompt: User prompt with context.
            response_model: Pydantic model class for response parsing.
            agent_role: Agent identifier for caching and logging.
            model: LLM model override (defaults to config default).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            use_cache: Whether to use response cache.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            LLMParsingError: After 3 retries with parse/validation failures.
        """
        cache_key = self._cache.hash_text(f"{agent_role}:{system}:{prompt}")

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    # Cache stores raw dict; handle both dict and list[dict]
                    data = cached if isinstance(cached, dict) else cached[0] if cached else {}
                    return response_model(**data)
                except (ValidationError, IndexError, TypeError):
                    pass  # Cache corrupted; regenerate

        async with self._semaphore:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    # Add JSON emphasis on retries
                    actual_prompt = prompt
                    if attempt > 0:
                        actual_prompt = prompt + "\n\n반드시 JSON만 출력하세요. 설명 없이 JSON 객체만."

                    raw = await self._call_api(
                        system=system,
                        prompt=actual_prompt,
                        model=model or self._default_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    parsed = self._parse_json(raw)
                    result = response_model(**parsed)

                    if use_cache:
                        self._cache.put(cache_key, [result.model_dump()])

                    return result

                except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
                    last_error = e
                    if attempt < 2:
                        wait = 2**attempt
                        logger.warning(
                            "Retry %d for %s (parse error, wait %ds): %s",
                            attempt + 1,
                            agent_role,
                            wait,
                            e,
                        )
                        await asyncio.sleep(wait)

            raise LLMParsingError(
                f"{agent_role}: failed to parse after 3 retries — {last_error}"
            ) from last_error

    async def _call_api(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Messages API (lazy client init)."""
        client = self._get_client()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response text.

        Handles markdown code blocks (```json ... ```) and raw JSON.
        Pattern reused from ClaudeVisionBackend._parse_response().
        """
        text = text.strip()

        # Try markdown code blocks first
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    parsed = json.loads(part)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and parsed:
                        return parsed[0] if isinstance(parsed[0], dict) else {"items": parsed}
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            return parsed[0] if isinstance(parsed[0], dict) else {"items": parsed}

        raise json.JSONDecodeError("Expected JSON object", text, 0)

    def _get_client(self) -> Any:
        """Lazy-init Anthropic async client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package required. Install: pip install anthropic"
                )
        return self._client
