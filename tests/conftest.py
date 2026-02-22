"""Shared test fixtures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Temporary working directory."""
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.fixture
def sample_config() -> dict:
    """Test configuration."""
    from reels.config import load_yaml_config
    return load_yaml_config()


@pytest.fixture
def mock_video_metadata() -> VideoMetadata:
    """Sample VideoMetadata fixture."""
    return VideoMetadata(
        source="test_video.mp4",
        duration_sec=15.0,
        fps=30.0,
        width=1080,
        height=1920,
        resolution="1080x1920",
        has_audio=True,
    )


@pytest.fixture
def mock_shots() -> list[Shot]:
    """Sample Shot list fixture (5 shots from a 15-second video)."""
    return [
        Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
        Shot(shot_id=1, start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, duration_sec=3.0),
        Shot(shot_id=2, start_sec=6.0, end_sec=9.5, start_frame=180, end_frame=285, duration_sec=3.5),
        Shot(shot_id=3, start_sec=9.5, end_sec=12.0, start_frame=285, end_frame=360, duration_sec=2.5),
        Shot(shot_id=4, start_sec=12.0, end_sec=15.0, start_frame=360, end_frame=450, duration_sec=3.0),
    ]


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Generate a 3-second synthetic test video with color transitions using ffmpeg.

    Returns None if ffmpeg is not available.
    """
    import shutil

    if not shutil.which("ffmpeg"):
        return None

    out = tmp_path_factory.mktemp("video") / "test.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=red:size=320x568:d=1:rate=30",
                "-f", "lavfi", "-i", "color=blue:size=320x568:d=1:rate=30",
                "-f", "lavfi", "-i", "color=green:size=320x568:d=1:rate=30",
                "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@pytest.fixture(scope="session")
def synthetic_audio(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Generate a 3-second synthetic test audio (sine wave) using ffmpeg."""
    import shutil

    if not shutil.which("ffmpeg"):
        return None

    out = tmp_path_factory.mktemp("audio") / "test.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "sine=frequency=440:duration=3:sample_rate=16000",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
