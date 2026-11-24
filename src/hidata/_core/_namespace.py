from __future__ import annotations

import warnings
from collections.abc import Callable


class _CachedAccessor:
    def __init__(self, name: str, accessor: type):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        accessor = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor)
        return accessor


def register_hidata_namespace(name: str) -> Callable[[type], type]:
    """Register a class as an extension namespace on :class:`HiData`.

    Packages that build on HiData, for example for genetics, patient representations or
    perturbation analysis, can expose their functions as ``hd.<name>.<function>()``. The decorated
    class is instantiated with the HiData on first access and cached on that object.

    Parameters
    ----------
    name
        Attribute name of the namespace.

    Returns
    -------
    Callable
        A class decorator.

    Examples
    --------
    >>> @hidata.register_hidata_namespace("qc")
    ... class QC:
    ...     def __init__(self, hd):
    ...         self._hd = hd
    ...
    ...     def small_groups(self, min_cells=100):
    ...         return self._hd.obs_names[self._hd.n_cells_per_group < min_cells]
    >>> hd.qc.small_groups()
    """
    from hidata._core.hidata import HiData

    def decorator(accessor: type) -> type:
        if hasattr(HiData, name):
            warnings.warn(f"Registering namespace '{name}' overrides an existing attribute of HiData.", stacklevel=2)
        setattr(HiData, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator
