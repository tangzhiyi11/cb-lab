"""统一异常系统 for cb-lab framework."""

from typing import Optional, Any


class CBLabException(Exception):
    """Base exception for cb-lab framework."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class ValidationError(CBLabException):
    """Raised when input validation fails."""

    pass


class RequestError(CBLabException):
    """Raised when request processing fails."""

    pass


class SchedulerError(CBLabException):
    """Raised when scheduler operations fail."""

    pass


class KVCacheError(CBLabException):
    """Raised when KV cache operations fail."""

    pass


class ModelError(CBLabException):
    """Raised when model operations fail."""

    pass


class ConfigurationError(CBLabException):
    """Raised when configuration is invalid."""

    pass


class MemoryError(CBLabException):
    """Raised when memory operations fail or limits are exceeded."""

    pass


class MonitoringError(CBLabException):
    """Raised when monitoring or profiling operations fail."""

    pass
