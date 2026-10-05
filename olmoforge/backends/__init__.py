"""Model backends. The mock backend is deterministic and trainable."""
from .mock import MockBackend
from .registry import register_backend, get_backend, list_backends

__all__ = ["MockBackend", "register_backend", "get_backend", "list_backends"]

from . import mock  # noqa: E402,F401  ensure registration runs
