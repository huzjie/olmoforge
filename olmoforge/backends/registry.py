"""Backend registry with a simple decorator."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def get_backend(name, **kwargs):
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available={list(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)


def list_backends():
    return sorted(_BACKENDS)
