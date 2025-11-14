"""Custom exceptions for Sharingan package."""


class SharinganError(Exception):
    """Base exception for Sharingan package."""
    pass


class VideoLoadError(SharinganError):
    """Raised when video cannot be loaded."""
    pass


class EncodingError(SharinganError):
    """Raised when frame encoding fails."""
    pass


class TemporalProcessingError(SharinganError):
    """Raised when temporal processing fails."""
    pass


class QueryError(SharinganError):
    """Raised when query execution fails."""
    pass
