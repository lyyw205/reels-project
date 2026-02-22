"""Custom exception hierarchy for reels pipeline."""


class ReelsError(Exception):
    """Base exception for reels package."""


class IngestError(ReelsError):
    """Ingest stage error."""


class DownloadError(IngestError):
    """Download failed."""


class AuthenticationRequiredError(DownloadError):
    """Authentication required for video."""


class VideoUnavailableError(DownloadError):
    """Video not found or unavailable."""


class NormalizationError(IngestError):
    """ffmpeg normalization failed."""


class DurationLimitError(IngestError):
    """Video exceeds maximum duration."""


class SegmentationError(ReelsError):
    """Shot segmentation error."""


class AnalysisError(ReelsError):
    """Analyzer error."""


class SynthesisError(ReelsError):
    """Template synthesis error."""


class DatabaseError(ReelsError):
    """Database error."""
