from __future__ import annotations

import numpy as np

from hidata._core.hidata import HiData


def _isin(values, wanted) -> np.ndarray:
    wanted = [wanted] if isinstance(wanted, str) or np.isscalar(wanted) else list(np.asarray(wanted).ravel())
    return np.asarray(values.isin(wanted))


def _apply(hd: HiData, selected: HiData, inplace: bool, copy: bool) -> HiData | None:
    if inplace:
        hd._finish(selected, inplace=True)
        return None
    return selected.copy() if copy else selected


def cell_level_obs_filter(
    hd: HiData,
    cell_level_key: str = None,
    cell_level_values=None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> HiData | None:
    """Keep cells whose ``C.obs[cell_level_key]`` is one of ``cell_level_values``.

    Groups that lose all their cells are dropped.

    Parameters
    ----------
    hd
        Data to filter.
    cell_level_key
        Column of ``C.obs``.
    cell_level_values
        A single value or several values to keep.
    inplace
        Modify ``hd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    HiData or None

    Examples
    --------
    >>> hidata.pp.cell_level_obs_filter(hd, "celltype", "CD8", inplace=False)
    """
    selected = hd.sel(C_obs=_isin(hd.C.obs[cell_level_key], cell_level_values))
    return _apply(hd, selected, inplace, copy)


def group_level_obs_filter(
    hd: HiData,
    group_level_key: str = None,
    group_level_values=None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> HiData | None:
    """Keep groups whose group table column ``group_level_key`` is one of ``group_level_values``.

    Parameters
    ----------
    hd
        Data to filter.
    group_level_key
        Column of the group table.
    group_level_values
        A single value or several values to keep.
    inplace
        Modify ``hd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    HiData or None

    Examples
    --------
    >>> hidata.pp.group_level_obs_filter(hd, "disease_state", "COVID-19", inplace=False)
    """
    selected = hd.sel(G_obs=_isin(hd.obs[group_level_key], group_level_values))
    return _apply(hd, selected, inplace, copy)


def group_level_var_filter(
    hd: HiData,
    *,
    group_level_key: str = None,
    group_level_values=None,
    inplace: bool = True,
    copy: bool = True,
) -> HiData | None:
    """Keep group variables whose ``G.var[group_level_key]`` is one of ``group_level_values``.

    Only available when ``G`` is an AnnData or MuData.

    Parameters
    ----------
    hd
        Data to filter.
    group_level_key
        Column of ``G.var``.
    group_level_values
        A single value or several values to keep.
    inplace
        Modify ``hd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    HiData or None

    Examples
    --------
    >>> hidata.pp.group_level_var_filter(hd, group_level_key="chrom", group_level_values="2", inplace=False)
    """
    if hd.G_type == "table":
        raise ValueError("G is a group table and has no variables to filter.")
    selected = hd.sel(G_var=_isin(hd.G.var[group_level_key], group_level_values))
    return _apply(hd, selected, inplace, copy)
