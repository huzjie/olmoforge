"""Configuration loading with a zero-dependency YAML fallback."""
import json
from pathlib import Path

from .utils.logging import get_logger

logger = get_logger("olmoforge.config")


class Config:
    """A thin attribute-accessible wrapper around a dict."""

    def __init__(self, data=None):
        object.__setattr__(self, "_d", data or {})

    def __getattr__(self, name):
        d = object.__getattribute__(self, "_d")
        if name in d:
            v = d[name]
            if isinstance(v, dict):
                return Config(v)
            return v
        raise AttributeError(name)

    def get(self, name, default=None):
        d = object.__getattribute__(self, "_d")
        v = d.get(name, default)
        if isinstance(v, dict):
            return Config(v)
        return v

    def to_dict(self):
        return dict(object.__getattribute__(self, "_d"))

    def __repr__(self):
        return f"Config({object.__getattribute__(self, '_d')!r})"


def _load_yaml(path):
    from .utils.yamlish import parse

    text = Path(path).read_text(encoding="utf-8")
    return parse(text)


def load_config(path):
    """Load a JSON or YAML config file. Falls back to the built-in YAML parser."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError:
            logger.info("PyYAML not installed; using built-in YAML subset parser")
            data = _load_yaml(path)
    return Config(data or {})
