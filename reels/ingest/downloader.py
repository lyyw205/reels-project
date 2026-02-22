"""Video download via yt-dlp."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import (
    AuthenticationRequiredError,
    DownloadError,
    VideoUnavailableError,
)

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Download videos from URLs using yt-dlp."""

    def __init__(self, output_dir: Path, config: dict | None = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or {}

    def download(self, source: str) -> Path:
        """Download video from URL or return local file path.

        Args:
            source: URL or local file path.

        Returns:
            Path to the downloaded/local video file.

        Raises:
            DownloadError: If download fails.
            AuthenticationRequiredError: If login is required.
            VideoUnavailableError: If video is not found.
        """
        if self._is_local_file(source):
            local = Path(source)
            if not local.exists():
                raise DownloadError(f"Local file not found: {source}")
            return local

        return self._download_with_ytdlp(source)

    def _is_local_file(self, source: str) -> bool:
        """Check if source is a local file path (not a URL)."""
        return not source.startswith(("http://", "https://", "www."))

    def _download_with_ytdlp(self, url: str) -> Path:
        """Download video using yt-dlp."""
        try:
            import yt_dlp
        except ImportError as e:
            raise DownloadError("yt-dlp is not installed") from e

        output_template = str(self.output_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise VideoUnavailableError(f"Could not extract info from: {url}")

                filename = ydl.prepare_filename(info)
                result_path = Path(filename)

                # yt-dlp may change extension after merge
                if not result_path.exists():
                    result_path = result_path.with_suffix(".mp4")

                if not result_path.exists():
                    raise DownloadError(f"Downloaded file not found at expected path: {result_path}")

                logger.info("Downloaded: %s → %s", url, result_path.name)
                return result_path

        except DownloadError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "login" in error_msg or "authentication" in error_msg or "private" in error_msg:
                raise AuthenticationRequiredError(f"Authentication required for: {url}") from e
            if "not found" in error_msg or "unavailable" in error_msg or "404" in error_msg:
                raise VideoUnavailableError(f"Video unavailable: {url}") from e
            raise DownloadError(f"Download failed for {url}: {e}") from e
