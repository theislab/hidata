from __future__ import annotations

import numpy as np
import pandas as pd

HIDATA_ENCODING_TYPE = "hidata"
HIDATA_ENCODING_VERSION = "0.1.0"
READABLE_ENCODING_VERSIONS = ("0.1.0",)
DONORDATA_ENCODING_TYPE = "donordata"
DONORDATA_ENCODING_VERSIONS = ("0.1.0", "0.2.0", "0.3.0")


class HiDataSchemaError(ValueError):
    """Raised by :func:`validate` when a HiData breaks one of its structural invariants."""


def _check_grouping(child_ids: pd.Series, parents: pd.Index, child: str, parent: str) -> None:
    ids = child_ids.astype(str)
    unknown = sorted(set(ids) - set(parents))
    if unknown:
        raise HiDataSchemaError(f"{child} refer to {parent} that are not present: {unknown[:8]}")
    codes = pd.Categorical(ids, categories=parents).codes
    empty = parents[np.bincount(codes, minlength=len(parents)) == 0].tolist()
    if empty:
        raise HiDataSchemaError(f"{parent.capitalize()} without any {child}: {empty[:8]}")
    if np.any(np.diff(codes) < 0):
        raise HiDataSchemaError(f"{child.capitalize()} are not grouped by {parent[:-1]} in the order of the {parent}.")


def validate(hd) -> bool:
    """Check that the levels of a HiData are consistent.

    Verifies that group and subgroup identifiers are unique, that every cell belongs to a known subgroup
    or group and every subgroup to a known group, that no group or subgroup is empty, and that cells and
    subgroups are grouped in the row order of the level above. When ``C.obs`` also names groups next to
    subgroups, both must agree. These are the invariants that construction and every public setter
    establish, so a failure means the object was modified through private attributes or in place.

    Parameters
    ----------
    hd
        The HiData to check.

    Returns
    -------
    bool
        ``True`` when every check passes.

    Raises
    ------
    HiDataSchemaError
        Naming the first check that failed.
    """
    groups = hd.obs_names
    if not groups.is_unique:
        duplicated = groups[groups.duplicated()].unique().tolist()
        raise HiDataSchemaError(f"Group identifiers in G are not unique: {duplicated[:8]}")
    if hd.S is None:
        _check_grouping(hd.C.obs[hd.group_id], groups, "cells", "groups")
        return True

    subgroups = hd.levels["subgroup"].obs_names
    if not subgroups.is_unique:
        duplicated = subgroups[subgroups.duplicated()].unique().tolist()
        raise HiDataSchemaError(f"Subgroup identifiers in S are not unique: {duplicated[:8]}")
    _check_grouping(hd.levels["subgroup"].obs[hd.group_id], groups, "subgroups", "groups")
    _check_grouping(hd.C.obs[hd.subgroup_id], subgroups, "cells", "subgroups")
    if hd.group_id in hd.C.obs.columns:
        subgroup_groups = hd.levels["subgroup"].obs[hd.group_id].astype(str).to_numpy()
        expected = subgroup_groups[pd.Categorical(hd.C.obs[hd.subgroup_id].astype(str), categories=subgroups).codes]
        if not np.array_equal(hd.C.obs[hd.group_id].astype(str).to_numpy(), expected):
            raise HiDataSchemaError("C.obs names groups that disagree with the groups of the subgroups.")
    return True
