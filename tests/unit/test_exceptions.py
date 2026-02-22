"""Tests for reels.exceptions module."""

from __future__ import annotations

from reels.exceptions import (
    AnalysisError,
    AuthenticationRequiredError,
    DatabaseError,
    DownloadError,
    DurationLimitError,
    IngestError,
    NormalizationError,
    ReelsError,
    SegmentationError,
    SynthesisError,
    VideoUnavailableError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        e = ReelsError("test")
        assert str(e) == "test"
        assert isinstance(e, Exception)

    def test_ingest_is_reels_error(self) -> None:
        assert issubclass(IngestError, ReelsError)

    def test_download_is_ingest_error(self) -> None:
        assert issubclass(DownloadError, IngestError)

    def test_auth_required_is_download_error(self) -> None:
        assert issubclass(AuthenticationRequiredError, DownloadError)

    def test_video_unavailable_is_download_error(self) -> None:
        assert issubclass(VideoUnavailableError, DownloadError)

    def test_normalization_is_ingest_error(self) -> None:
        assert issubclass(NormalizationError, IngestError)

    def test_duration_limit_is_ingest_error(self) -> None:
        assert issubclass(DurationLimitError, IngestError)

    def test_segmentation_is_reels_error(self) -> None:
        assert issubclass(SegmentationError, ReelsError)

    def test_analysis_is_reels_error(self) -> None:
        assert issubclass(AnalysisError, ReelsError)

    def test_synthesis_is_reels_error(self) -> None:
        assert issubclass(SynthesisError, ReelsError)

    def test_database_is_reels_error(self) -> None:
        assert issubclass(DatabaseError, ReelsError)

    def test_catch_all_with_reels_error(self) -> None:
        for exc_cls in (IngestError, DownloadError, SegmentationError, AnalysisError, SynthesisError, DatabaseError):
            try:
                raise exc_cls("msg")
            except ReelsError:
                pass  # expected
