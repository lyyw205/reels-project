"""Tests for reels.production.creative_team.llm.CreativeLLM."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from reels.production.creative_team.llm import CreativeLLM, LLMParsingError


class _TestModel(BaseModel):
    name: str
    value: int


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_llm(tmp_path) -> CreativeLLM:
    """Create a CreativeLLM with a tmp cache dir."""
    config = {"production": {"creative_team": {"cache_dir": str(tmp_path / "cache")}}}
    return CreativeLLM(config=config)


def _mock_response(text: str):
    """Build a mock Anthropic response object."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ─── _parse_json ─────────────────────────────────────────────────────


def test_parse_json_plain_json(tmp_path):
    llm = _make_llm(tmp_path)
    data = llm._parse_json('{"name": "test", "value": 42}')
    assert data == {"name": "test", "value": 42}


def test_parse_json_markdown_code_block(tmp_path):
    llm = _make_llm(tmp_path)
    text = '```json\n{"name": "hello", "value": 7}\n```'
    data = llm._parse_json(text)
    assert data["name"] == "hello"
    assert data["value"] == 7


def test_parse_json_markdown_without_language_tag(tmp_path):
    llm = _make_llm(tmp_path)
    text = '```\n{"name": "world", "value": 1}\n```'
    data = llm._parse_json(text)
    assert data["name"] == "world"


def test_parse_json_invalid_raises(tmp_path):
    llm = _make_llm(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        llm._parse_json("this is not json at all")


def test_parse_json_list_returns_first_dict(tmp_path):
    llm = _make_llm(tmp_path)
    text = '[{"name": "first", "value": 1}, {"name": "second", "value": 2}]'
    data = llm._parse_json(text)
    assert data == {"name": "first", "value": 1}


# ─── generate() success ──────────────────────────────────────────────


def test_generate_success(tmp_path):
    """generate() calls _call_api once and returns parsed model."""
    llm = _make_llm(tmp_path)

    valid_json = '{"name": "숙소", "value": 99}'

    async def _run():
        llm._call_api = AsyncMock(return_value=valid_json)
        result = await llm.generate(
            system="sys",
            prompt="prompt",
            response_model=_TestModel,
            agent_role="test_agent",
            use_cache=False,
        )
        return result

    result = asyncio.run(_run())
    assert isinstance(result, _TestModel)
    assert result.name == "숙소"
    assert result.value == 99


def test_generate_calls_call_api_exactly_once_on_success(tmp_path):
    llm = _make_llm(tmp_path)
    valid_json = '{"name": "ok", "value": 1}'

    async def _run():
        llm._call_api = AsyncMock(return_value=valid_json)
        await llm.generate(
            system="s",
            prompt="p",
            response_model=_TestModel,
            agent_role="test",
            use_cache=False,
        )
        return llm._call_api.call_count

    count = asyncio.run(_run())
    assert count == 1


# ─── generate() retry on parse error ────────────────────────────────


def test_generate_retries_on_parse_error_then_succeeds(tmp_path):
    """First 2 calls return garbage; 3rd returns valid JSON → success."""
    llm = _make_llm(tmp_path)
    valid_json = '{"name": "retry_ok", "value": 5}'

    async def _run():
        llm._call_api = AsyncMock(
            side_effect=["not json", "also bad", valid_json]
        )
        # Patch sleep to avoid waiting in tests
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await llm.generate(
                system="s",
                prompt="p",
                response_model=_TestModel,
                agent_role="test",
                use_cache=False,
            )
        return result, llm._call_api.call_count

    result, count = asyncio.run(_run())
    assert isinstance(result, _TestModel)
    assert result.name == "retry_ok"
    assert count == 3


def test_generate_raises_llm_parsing_error_after_3_failures(tmp_path):
    """All 3 attempts return garbage → LLMParsingError."""
    llm = _make_llm(tmp_path)

    async def _run():
        llm._call_api = AsyncMock(side_effect=["bad", "bad", "bad"])
        with patch("asyncio.sleep", new=AsyncMock()):
            await llm.generate(
                system="s",
                prompt="p",
                response_model=_TestModel,
                agent_role="test",
                use_cache=False,
            )

    with pytest.raises(LLMParsingError):
        asyncio.run(_run())


def test_generate_raises_llm_parsing_error_on_validation_failure(tmp_path):
    """Valid JSON but wrong schema → LLMParsingError after 3 retries."""
    llm = _make_llm(tmp_path)
    # Missing required 'name' and 'value' fields
    bad_json = '{"wrong_field": "x"}'

    async def _run():
        llm._call_api = AsyncMock(side_effect=[bad_json, bad_json, bad_json])
        with patch("asyncio.sleep", new=AsyncMock()):
            await llm.generate(
                system="s",
                prompt="p",
                response_model=_TestModel,
                agent_role="test",
                use_cache=False,
            )

    with pytest.raises(LLMParsingError):
        asyncio.run(_run())


# ─── Caching ─────────────────────────────────────────────────────────


def test_generate_caches_result(tmp_path):
    """First call hits API and caches; second call uses cache (no API call)."""
    llm = _make_llm(tmp_path)
    valid_json = '{"name": "cached", "value": 42}'

    async def _run():
        llm._call_api = AsyncMock(return_value=valid_json)

        # First call: cache miss → calls API
        r1 = await llm.generate(
            system="sys",
            prompt="prompt",
            response_model=_TestModel,
            agent_role="test",
            use_cache=True,
        )
        first_call_count = llm._call_api.call_count

        # Second call: same inputs → cache hit → no additional API call
        r2 = await llm.generate(
            system="sys",
            prompt="prompt",
            response_model=_TestModel,
            agent_role="test",
            use_cache=True,
        )
        second_call_count = llm._call_api.call_count

        return r1, r2, first_call_count, second_call_count

    r1, r2, first, second = asyncio.run(_run())
    assert r1.name == "cached"
    assert r2.name == "cached"
    assert first == 1      # API called once
    assert second == 1     # Cache hit: no additional call


def test_generate_use_cache_false_skips_cache(tmp_path):
    """use_cache=False always calls API even with same inputs."""
    llm = _make_llm(tmp_path)
    valid_json = '{"name": "nocache", "value": 0}'

    async def _run():
        llm._call_api = AsyncMock(return_value=valid_json)
        await llm.generate(
            system="s", prompt="p", response_model=_TestModel,
            agent_role="t", use_cache=False,
        )
        await llm.generate(
            system="s", prompt="p", response_model=_TestModel,
            agent_role="t", use_cache=False,
        )
        return llm._call_api.call_count

    count = asyncio.run(_run())
    assert count == 2


# ─── Retry prompt augmentation ───────────────────────────────────────


def test_generate_augments_prompt_on_retry(tmp_path):
    """On retry attempts, prompt is augmented with JSON-only instruction."""
    llm = _make_llm(tmp_path)
    valid_json = '{"name": "aug", "value": 1}'
    call_prompts: list[str] = []

    async def _fake_call_api(**kwargs):
        call_prompts.append(kwargs["prompt"])
        if len(call_prompts) < 3:
            return "bad json"
        return valid_json

    async def _run():
        llm._call_api = _fake_call_api
        with patch("asyncio.sleep", new=AsyncMock()):
            await llm.generate(
                system="s",
                prompt="original",
                response_model=_TestModel,
                agent_role="test",
                use_cache=False,
            )
        return call_prompts

    prompts = asyncio.run(_run())
    assert prompts[0] == "original"
    assert "JSON만" in prompts[1]  # augmented on retry
    assert "JSON만" in prompts[2]
