"""DREAM-Jev: replay-based meta-exploration for Jev Ultrafast.

This module intentionally improves only bounded exploration policy parameters.
The browser executor, approval policy, verifier contract, and isolation boundary
remain outside the self-improvement surface.
"""

import sys as _sys
import types as _types
from typing import Iterable  # noqa: F401  (facade: re-exported module attr)

from jev_ultrafast import _dream as _impl_pkg
from jev_ultrafast._dream import *  # noqa: F401,F403
from jev_ultrafast._dream import __all__ as __all__

# Compatibility: before the internal split, every implementation name and
# every imported module lived in this module's namespace, so callers could
# monkeypatch e.g. ``dream.time`` or ``dream.ReplaySimulator`` and have the
# implementation observe the patch. Re-expose every name bound in any
# implementation submodule here and forward attribute writes back to every
# submodule that binds the name, preserving that behavior.
_OWNERS = {}
for _mod in _impl_pkg._MODULES:
    for _n in vars(_mod):
        _OWNERS.setdefault(_n, []).append(_mod)

_self = _sys.modules[__name__]
for _n, _owners in _OWNERS.items():
    if _n not in _self.__dict__:
        _self.__dict__[_n] = getattr(_owners[0], _n)


class _FacadeModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for _owner in _OWNERS.get(name, ()):
            setattr(_owner, name, value)


_self.__class__ = _FacadeModule
_self.__dict__.pop("TYPE_CHECKING", None)
del _sys, _types, _impl_pkg, _n, _owners, _mod, _self, _FacadeModule
