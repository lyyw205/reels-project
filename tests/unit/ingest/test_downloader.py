"""Tests for reels.ingest.downloader module."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.exceptions import DownloadError
from reels.ingest.downloader import VideoDownloader


class TestVideoDownloader:
    def test_local_file_exists(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        dl = VideoDownloader(tmp_path / "downloads")
        result = dl.download(str(video))
        assert result == video

    def test_local_file_not_found(self, tmp_path: Path) -> None:
        dl = VideoDownloader(tmp_path / "downloads")
        with pytest.raises(DownloadError, match="not found"):
            dl.download("/nonexistent/path/video.mp4")

    def test_is_local_file_detection(self, tmp_path: Path) -> None:
        dl = VideoDownloader(tmp_path)
        assert dl._is_local_file("/path/to/video.mp4") is True
        assert dl._is_local_file("relative/video.mp4") is True
        assert dl._is_local_file("https://example.com/video") is False
        assert dl._is_local_file("http://example.com/video") is False
        assert dl._is_local_file("www.example.com/video") is False

    def test_output_dir_created(self, tmp_path: Path) -> None:
        dl = VideoDownloader(tmp_path / "new" / "nested" / "dir")
        assert (tmp_path / "new" / "nested" / "dir").is_dir()
