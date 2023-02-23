

class Proxy:
    """Base class for all wrappers."""

    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)

    def __dir__(self):
        return list(set(super().__dir__()) | set(dir(self._shadowed)))
