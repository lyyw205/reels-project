"""Tests for the CLI produce command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from reels.cli import cli
from reels.production.models import (
    ClaimLevel,
    FeatureCategory,
    ProductionResult,
    Storyboard,
    StoryboardShot,
    ShotRole,
    VerifiedFeature,
)
from reels.models.template import CameraType


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_images(tmp_path: Path) -> list[str]:
    """Create temporary image files."""
    imgs = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"fake image data")
        imgs.append(str(p))
    return imgs


def _make_feature(tag: str = "노천탕") -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag,
        tag_en="outdoor_bath",
        confidence=0.8,
        evidence_images=[],
        category=FeatureCategory.AMENITY,
        claim_level=ClaimLevel.CONFIRMED,
    )


def _make_result(status: str = "complete", n_features: int = 3, n_shots: int = 7) -> ProductionResult:
    storyboard = None
    if status == "complete":
        storyboard = MagicMock(spec=Storyboard)
        storyboard.shots = [MagicMock() for _ in range(n_shots)]
        storyboard.total_duration_sec = 12.5
        storyboard.project_id = "shorts_abc12345"

    return ProductionResult(
        project_id="shorts_abc12345",
        status=status,
        storyboard=storyboard,
        features=[_make_feature(f"feat_{i}") for i in range(n_features)],
        errors=["No features found"] if status == "failed" else [],
    )


class TestProduceCommand:
    def test_produce_no_images_shows_error(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["produce"])
        assert result.exit_code != 0

    @patch("reels.production.agent.ProductionAgent")
    def test_produce_success(self, mock_agent_cls, runner: CliRunner, fake_images: list[str]) -> None:
        mock_agent = MagicMock()
        mock_agent.produce = AsyncMock(return_value=_make_result())
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["produce", *fake_images, "--name", "호텔A"])
        assert result.exit_code == 0
        assert "Production complete" in result.output
        assert "shorts_abc12345" in result.output

    @patch("reels.production.agent.ProductionAgent")
    def test_produce_failed_result(self, mock_agent_cls, runner: CliRunner, fake_images: list[str]) -> None:
        mock_agent = MagicMock()
        mock_agent.produce = AsyncMock(return_value=_make_result(status="failed"))
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["produce", *fake_images])
        assert result.exit_code != 0

    @patch("reels.production.agent.ProductionAgent")
    def test_produce_no_web_flag(self, mock_agent_cls, runner: CliRunner, fake_images: list[str]) -> None:
        mock_agent = MagicMock()
        mock_agent.produce = AsyncMock(return_value=_make_result())
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["produce", *fake_images, "--no-web"])
        assert result.exit_code == 0
        # Verify config was passed with web disabled
        call_kwargs = mock_agent_cls.call_args
        config = call_kwargs[1].get("config") or call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("config", {})
        if isinstance(config, dict):
            assert config.get("production", {}).get("web_verification", {}).get("enabled") is False

    @patch("reels.production.agent.ProductionAgent")
    def test_produce_with_target(self, mock_agent_cls, runner: CliRunner, fake_images: list[str]) -> None:
        mock_agent = MagicMock()
        mock_agent.produce = AsyncMock(return_value=_make_result())
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["produce", *fake_images, "--target", "family"])
        assert result.exit_code == 0

    @patch("reels.production.agent.ProductionAgent")
    def test_produce_output_shows_details(self, mock_agent_cls, runner: CliRunner, fake_images: list[str]) -> None:
        mock_agent = MagicMock()
        mock_agent.produce = AsyncMock(return_value=_make_result())
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["produce", *fake_images])
        assert "Features: 3" in result.output
        assert "Shots: 7" in result.output
        assert "Duration: 12.5s" in result.output
