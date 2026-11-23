from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from hidata._core.hidata import HiData


def _stack_rows(values: list, join: str):
    if all(isinstance(v, pd.DataFrame) for v in values):
        return pd.concat(values, axis=0, join=join)
    if any(sp.issparse(v) for v in values):
        return sp.vstack([sp.csr_array(v) for v in values])
    return np.concatenate([np.asarray(v) for v in values], axis=0)


def _concat_level(containers: list, is_table: bool, key: str, label, keys, join):
    """Concatenate one level of several inputs, returning the object and any table matrices."""
    if is_table:
        table = pd.concat([c.obs for c in containers], axis=0, join=join)
        table.index.name = key
        if label is not None:
            table[label] = pd.Categorical(np.repeat(keys, [c.n_obs for c in containers]), categories=keys)
        obsm = {}
        for name in sorted(set.intersection(*(set(c.obsm.keys()) for c in containers))):
            values = [c.obsm[name] for c in containers]
            if len({v.shape[1] for v in values}) == 1 or all(isinstance(v, pd.DataFrame) for v in values):
                obsm[name] = _stack_rows(values, join)
        return table, obsm
    adata = ad.concat(containers, join=join, merge="same", label=label, keys=keys if label else None)
    adata.obs.index.name = key
    return adata, None


def concat(
    hds: Sequence[HiData] | Mapping[str, HiData],
    *,
    label: str | None = None,
    join: Literal["inner", "outer"] = "inner",
    index_unique: str | None = None,
) -> HiData:
    """Concatenate cohorts along the group axis.

    Group and subgroup identifiers must be unique across the inputs. Cells are concatenated with
    :func:`anndata.concat`. Group- and subgroup-aligned matrices present in every input with matching
    widths are stacked. Pairwise matrices are dropped because they are undefined across cohorts.

    Parameters
    ----------
    hds
        HiData objects with the same identifiers and the same kind of ``G`` and ``S``, either all
        tables or all AnnData. A mapping uses its keys as cohort labels.
    label
        Group table column that records the cohort of every group. Requires a mapping, or uses the
        position of each input otherwise.
    join
        How to combine the variables of ``C`` and of AnnData levels.
    index_unique
        Separator used to make cell names unique by appending the cohort label, see
        :func:`anndata.concat`.

    Returns
    -------
    HiData
    """
    if isinstance(hds, Mapping):
        keys, hds = list(hds.keys()), list(hds.values())
    else:
        hds = list(hds)
        keys = [str(i) for i in range(len(hds))]
    if not hds:
        raise ValueError("Nothing to concatenate.")
    group_id, subgroup_id = hds[0].group_id, hds[0].subgroup_id
    if any(hd.group_id != group_id or hd.subgroup_id != subgroup_id for hd in hds):
        raise ValueError("All inputs must share the same group_id and subgroup_id.")
    kinds = {hd.G_type for hd in hds}
    if len(kinds) != 1 or kinds == {"mudata"}:
        raise ValueError(f"Cannot concatenate G of kinds {sorted(kinds)}, use all tables or all AnnData.")
    if subgroup_id is not None and len({hd._S_is_table for hd in hds}) != 1:
        raise ValueError("Cannot concatenate subgroup levels of different kinds, use all tables or all AnnData.")
    if any(not isinstance(hd.C, AnnData) for hd in hds):
        raise ValueError("concat requires C to be an AnnData in every input.")
    if any(type(level).__name__ == "EHRData" for hd in hds for level in (hd._G, hd._S)):
        raise ValueError("concat cannot combine EHRData levels, their time axes would be lost.")

    for level in ("group", "subgroup") if subgroup_id is not None else ("group",):
        names = pd.Index(np.concatenate([hd.levels[level].obs_names.to_numpy() for hd in hds]))
        if not names.is_unique:
            duplicated = names[names.duplicated()].unique()[:8].tolist()
            raise ValueError(f"{level.capitalize()} identifiers are not unique across inputs: {duplicated}")

    G, obsm = _concat_level([hd._G for hd in hds], kinds == {"table"}, group_id, label, keys, join)
    S, subgroup_obsm = None, None
    if subgroup_id is not None:
        S, subgroup_obsm = _concat_level([hd._S for hd in hds], hds[0]._S_is_table, subgroup_id, None, keys, join)
    C = ad.concat(
        [hd.C for hd in hds],
        join=join,
        merge="same",
        keys=keys if index_unique else None,
        index_unique=index_unique,
    )
    uns = {}
    for hd in reversed(hds):
        uns.update(hd.uns)
    synced = [k for k in hds[0]._var_dims_to_sync if all(k in hd._var_dims_to_sync for hd in hds)]
    merged = HiData(
        G=G, S=S, C=C, group_id=group_id, subgroup_id=subgroup_id, obsm=obsm, uns=uns, var_dims_to_sync=synced
    )
    if subgroup_obsm:
        positions = S.index.get_indexer(merged._S.obs_names)
        for name, value in subgroup_obsm.items():
            merged._S.obsm[name] = value.iloc[positions] if isinstance(value, pd.DataFrame) else value[positions]
    return merged
