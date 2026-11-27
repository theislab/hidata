from __future__ import annotations

import contextlib
import copy as _copy
import logging
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from mudata import MuData

from hidata._core import _utils
from hidata._core._repr import render
from hidata._core._write import is_ehrdata, reads_from, write_group
from hidata._core.fields import GAnn

logger = logging.getLogger(__name__)


def _log_missing_labels(labels: pd.Series, units: str, key: str, shown: int = 10, missing=None) -> None:
    """Warn which units are dropped because their ``key`` is missing or a string such as ``"nan"``."""
    dropped = labels[_utils.is_missing_label(labels) if missing is None else missing]
    if dropped.empty:
        return
    names = ", ".join(
        repr(name) if str(name) == str(value) else f"{name!r} ({value!r})"
        for name, value in dropped.iloc[:shown].items()
    )
    more = f" and {len(dropped) - shown} more" if len(dropped) > shown else ""
    logger.warning(
        f"Dropping {len(dropped)} {units} with a missing '{key}' (empty, or a value such as NaN, None, 'nan' or 'NA'): "
        f"{names}{more}. If these are real names, rename them first."
    )


MISSING_GROUP = "nan"

Selection = str | Callable | Sequence | np.ndarray | pd.Series | pd.Index
Level = Literal["group", "subgroup", "cell"]
_RANK = {"group": 0, "subgroup": 1, "cell": 2}


def _keep_categories(keep: bool = True):
    """Context in which views of a group-level AnnData keep all categories of ``obs`` and ``var``.

    Without it anndata drops unused categories of every categorical column whenever a view is
    created, also of ``var`` when only groups are selected, which for genotypes (alleles of millions
    of variants) costs more than reading the selected genotypes.
    """
    return ad.settings.override(remove_unused_categories=False) if keep else contextlib.nullcontext()


class HiData:
    """A hierarchical data object.

    :class:`~hidata.HiData` stores single-cell data :attr:`C` together with data on the groups
    the cells come from, :attr:`G`, and optionally on the subgroups in between, :attr:`S`. ``G`` holds
    one row per group and can be a DataFrame of group covariates, an :class:`~anndata.AnnData` such as
    genotypes, bulk omics or clinical measurements, an EHRData with a time axis, or a
    :class:`~mudata.MuData`. ``S`` holds one row per subgroup, such as a sample, time point, tissue or replicate,
    and its ``group_id`` column names the group of every subgroup. ``C`` holds the cells as an AnnData or
    MuData, with the subgroup of every cell in ``C.obs[subgroup_id]``, or its group in
    ``C.obs[group_id]`` when there is no subgroup level.

    On construction, and whenever a level is replaced, only units connected across all levels are
    kept: groups with subgroups, subgroups with cells and cells with a known subgroup or group. Subgroups are
    grouped by group in the row order of ``G`` and cells by subgroup, or by group, in the row order of
    the level above. Matrices aligned to groups or subgroups, such as embeddings, compositions,
    pseudobulk profiles or distances, live in ``obsm`` and ``obsp`` of their level and are subset
    with it.

    A group can be any unit that groups cells, for example a donor or patient, a target gene in a
    CRISPR screen or a compound in a drug screen. A subgroup is optional, for example a sample, a time
    point, a guide or a dose.

    Parameters
    ----------
    G
        Group-level data with one row per group. Group identifiers are taken from the ``group_id``
        column when there is one and from the index otherwise. Optional when ``S`` is given, in which
        case it is a table of the groups named in ``S[group_id]`` without further columns.
    C
        Cell-level data as an AnnData or MuData. ``C.obs`` holds the subgroup of every cell in
        ``subgroup_id`` when ``S`` is given, and the group of every cell in ``group_id`` otherwise.
    group_id
        Name of the group identifier.
    S
        Optional subgroup-level data with one row per subgroup and the group of every subgroup in a
        ``group_id`` column. Subgroup identifiers are taken from the ``subgroup_id`` column when there is
        one and from the index otherwise.
    subgroup_id
        Name of the subgroup identifier. Required with ``S``.
    var_dims_to_sync
        Keys of group-level ``obsm`` whose columns are kept in sync with ``C.var_names``.
    uns
        Unstructured annotations of the object.
    obsm
        Group-aligned matrices to add to :attr:`obsm`, with rows in the row order of ``G``.
    obsp
        Group-by-group matrices to add to :attr:`obsp`, with rows and columns in the row order of ``G``.

    Examples
    --------
    >>> hd = HiData(G=patients, C=adata, group_id="patient")
    >>> hd = HiData(S=samples, C=adata, group_id="patient", subgroup_id="sample")
    >>> hd = HiData.from_anndata(adata, group_id="patient", subgroup_id="sample")
    >>> hd.filter_groups("age > 60").filter_groups_by_cells(20, groupby="cell_type")
    >>> hd.get_df(["age", "cell_type", "CD14"], level="cell")
    """

    def __init__(
        self,
        *,
        G: pd.DataFrame | AnnData | MuData | None = None,
        C: AnnData | MuData,
        group_id: str = GAnn.group,
        S: pd.DataFrame | AnnData | None = None,
        subgroup_id: str | None = None,
        var_dims_to_sync: list[str] | None = None,
        uns: dict | None = None,
        obsm: dict | None = None,
        obsp: dict | None = None,
    ):
        if (S is None) != (subgroup_id is None):
            raise TypeError("S and subgroup_id must be given together.")
        if G is None and S is None:
            raise TypeError(
                "Pass the group-level data as G, or a subgroup table S that names the group of every subgroup."
            )
        if not isinstance(C, AnnData | MuData):
            raise TypeError(f"C must be an AnnData or MuData object, got {type(C).__name__}.")
        if is_ehrdata(C):
            raise TypeError("C must be an AnnData or MuData object; an EHRData belongs to G or S.")
        self.group_id = group_id
        self.subgroup_id = subgroup_id
        cell_key = subgroup_id if S is not None else group_id
        if cell_key not in C.obs.columns:
            raise ValueError(f"'{cell_key}' not found in C.obs")
        if S is not None:
            S, S_is_table = self._prepare_level(S, subgroup_id)
            if group_id not in S.obs.columns:
                raise ValueError(
                    f"'{group_id}' not found in the subgroup table, it must name the group of every subgroup."
                )
        else:
            S_is_table = False
        if G is None:
            names = pd.unique(pd.Series(_utils.string_labels(S.obs[group_id])).dropna())
            G = pd.DataFrame(index=pd.Index(np.asarray(names, dtype=str), name=group_id))
        given = G
        G, G_is_table = self._prepare_level(G, group_id)
        if (obsm or obsp) and G is given:
            G = _detached(G)
        for key, value in (obsm or {}).items():
            G.obsm[key] = value
        for key, value in (obsp or {}).items():
            G.obsp[key] = value
        self._var_dims_to_sync = [] if var_dims_to_sync is None else list(var_dims_to_sync)
        self.uns = {} if uns is None else uns
        self._match(G, S, C, G_is_table, S_is_table)

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData | MuData,
        group_id: str = GAnn.group,
        *,
        subgroup_id: str | None = None,
        columns: Sequence[str] | None = None,
        keep_in_cells: bool = False,
    ) -> HiData:
        """Split a cell-level object into tables for groups, optionally subgroups, and the cells.

        Columns of ``adata.obs`` that take a single value within every group are moved into a group
        table, which becomes ``G``. With ``subgroup_id``, columns that are constant within subgroups but
        not within groups go to a subgroup table ``S``. This turns metadata that is repeated on every
        cell into one row per unit, so it can be filtered, edited and stored once.

        Parameters
        ----------
        adata
            Cell-level data with the group of every cell in ``adata.obs[group_id]``.
        group_id
            Column of ``adata.obs`` with group identifiers.
        subgroup_id
            Column of ``adata.obs`` with subgroup identifiers. Every subgroup must belong to one group.
        columns
            Columns to move. By default every column that is constant within groups, or within
            subgroups. Columns given explicitly must be constant within subgroups, or within groups
            without ``subgroup_id``.
        keep_in_cells
            Keep the moved columns in ``C.obs`` as well. Without it ``C`` is a shallow copy of
            ``adata`` with the columns removed, sharing all matrices with ``adata``.

        Returns
        -------
        HiData
            With units sorted by their identifier, or in category order for categorical columns.
        """
        obs = adata.obs
        for key in (group_id, subgroup_id):
            if key is not None and key not in obs.columns:
                raise ValueError(f"'{key}' not found in adata.obs")
        group_labels = pd.Series(_utils.string_labels(obs[group_id]))
        ids = {group_id} if subgroup_id is None else {group_id, subgroup_id}
        if subgroup_id is not None:
            unit_codes, subgroups = pd.factorize(pd.Series(_utils.string_labels(obs[subgroup_id])), sort=True)
            unit_first = _utils.first_positions(unit_codes, len(subgroups))
            known = group_labels.notna().to_numpy() & (unit_codes >= 0)
            per_subgroup = group_labels[known].groupby(unit_codes[known])
            if (per_subgroup.nunique() > 1).any():
                raise ValueError(f"Some subgroups in '{subgroup_id}' contain cells of several groups.")
            group_labels = group_labels.fillna(pd.Series(unit_codes).map(per_subgroup.first()))
        group_codes, groups = pd.factorize(group_labels, sort=True)
        group_first = _utils.first_positions(group_codes, len(groups))
        if subgroup_id is None:
            unit_codes, unit_first = group_codes, group_first

        if columns is None:
            candidates = [c for c in obs.columns if c not in ids]
            unit_columns = [c for c in candidates if _utils.is_constant_within(obs[c], unit_codes, unit_first)]
        else:
            unit_columns = [c for c in columns if c not in ids]
            varying = [c for c in unit_columns if not _utils.is_constant_within(obs[c], unit_codes, unit_first)]
            if varying:
                level = "subgroups" if subgroup_id is not None else "groups"
                raise ValueError(f"Columns {varying} vary within {level} and cannot be moved out of the cells.")

        if subgroup_id is None:
            group_columns, subgroup_columns = unit_columns, []
        else:
            group_columns = [c for c in unit_columns if _utils.is_constant_within(obs[c], group_codes, group_first)]
            subgroup_columns = [c for c in unit_columns if c not in group_columns]

        table = obs.iloc[group_first][list(group_columns)].copy()
        table.index = pd.Index(np.asarray(groups, dtype=str), name=group_id)
        moved = list(group_columns) + list(subgroup_columns)
        cells = adata if keep_in_cells or not moved else _utils.shallow_with_obs(adata, obs.drop(columns=moved))
        if subgroup_id is None:
            return cls(G=table, C=cells, group_id=group_id)
        subgroup_table = obs.iloc[unit_first][[group_id, *subgroup_columns]].copy()
        subgroup_table[group_id] = np.asarray(group_labels.iloc[unit_first], dtype=object)
        subgroup_table.index = pd.Index(np.asarray(subgroups, dtype=str), name=subgroup_id)
        return cls(G=table, S=subgroup_table, C=cells, group_id=group_id, subgroup_id=subgroup_id)

    def _prepare_level(self, value, key: str) -> tuple[AnnData | MuData, bool]:
        if isinstance(value, pd.DataFrame):
            return AnnData(obs=_indexed_by(value, key, "a column of the table or the name of its index")), True
        if not isinstance(value, AnnData | MuData):
            raise TypeError(f"Expected a DataFrame, AnnData or MuData object, got {type(value).__name__}.")
        obs = value.obs
        if isinstance(value, MuData) and obs.index.name is None:
            if any(mod.obs.index.name == key for mod in value.mod.values()):
                obs = obs.rename_axis(key)
        obs = _indexed_by(obs, key, "in obs or set as its index")
        if obs.index.name == value.obs.index.name and obs.equals(value.obs):
            return value, False
        return _detached(value, obs), False

    def _match(self, G, S, C, G_is_table: bool, S_is_table: bool, checked: bool = False) -> None:
        if not checked:
            with _keep_categories(not G_is_table):
                G = _drop_missing_names(G, "groups", self.group_id)
        if not G.obs_names.is_unique:
            raise ValueError("Group identifiers in G must be unique.")
        if S is None:
            cell_codes = _match_codes(C.obs[self.group_id], G.obs_names, "cells", self.group_id, checked)
            present = np.bincount(cell_codes[cell_codes >= 0], minlength=G.n_obs) > 0
            if not present.any():
                self._raise_empty(G, None, C)
            if not present.all():
                with _keep_categories(not G_is_table):
                    G = G[present]
                new_position = np.cumsum(present) - 1
                cell_codes = np.where(cell_codes >= 0, new_position[np.maximum(cell_codes, 0)], -1)
        else:
            if not checked:
                S = _drop_missing_names(S, "subgroups", self.subgroup_id)
            if not S.obs_names.is_unique:
                raise ValueError("Subgroup identifiers in S must be unique.")
            subgroup_groups = _match_codes(S.obs[self.group_id], G.obs_names, "subgroups", self.group_id, checked)
            cell_codes = _match_codes(C.obs[self.subgroup_id], S.obs_names, "cells", self.subgroup_id, checked)
            linked = cell_codes >= 0
            linked[linked] = subgroup_groups[cell_codes[linked]] >= 0
            cell_codes = np.where(linked, cell_codes, -1)
            subgroup_present = (np.bincount(cell_codes[linked], minlength=S.n_obs) > 0) & (subgroup_groups >= 0)
            group_present = np.bincount(subgroup_groups[subgroup_present], minlength=G.n_obs) > 0
            if not group_present.any():
                self._raise_empty(G, S, C)
            if not group_present.all():
                with _keep_categories(not G_is_table):
                    G = G[group_present]
                new_position = np.cumsum(group_present) - 1
                subgroup_groups = np.where(subgroup_groups >= 0, new_position[np.maximum(subgroup_groups, 0)], -1)
            subgroups = np.flatnonzero(subgroup_present)
            subgroup_order = subgroups[np.argsort(subgroup_groups[subgroups], kind="stable")]
            if subgroup_order.size != S.n_obs or not np.array_equal(subgroup_order, np.arange(S.n_obs)):
                S = S[subgroup_order]
            new_position = np.full(subgroup_present.size, -1)
            new_position[subgroup_order] = np.arange(subgroup_order.size)
            cell_codes = np.where(cell_codes >= 0, new_position[np.maximum(cell_codes, 0)], -1)
            if S_is_table and S.is_view:
                S = S.copy()
        if G_is_table and G.is_view:
            G = G.copy()

        cells = np.flatnonzero(cell_codes >= 0)
        order = cells[np.argsort(cell_codes[cells], kind="stable")]
        if order.size != C.n_obs or not np.array_equal(order, np.arange(C.n_obs)):
            C = C[order]

        if S is not None and self.group_id in C.obs.columns:
            subgroup_groups = G.obs_names[_utils.codes_for(S.obs[self.group_id], G.obs_names)].to_numpy()
            expected = subgroup_groups[_utils.codes_for(C.obs[self.subgroup_id], S.obs_names)]
            stated = np.asarray(_utils.string_labels(C.obs[self.group_id]), dtype=object)
            known = ~pd.isna(stated)
            if not np.array_equal(stated[known].astype(str), expected[known].astype(str)):
                raise ValueError(f"C.obs['{self.group_id}'] disagrees with the group of the subgroup of some cells.")

        self._G, self._S, self._C = G, S, C
        self._G_is_table, self._S_is_table = G_is_table, S_is_table

    def _raise_empty(self, G, S, C) -> None:
        if G.n_obs == 0 or C.n_obs == 0 or (S is not None and S.n_obs == 0):
            raise ValueError("The selection leaves no groups with cells.")
        if self.subgroup_id is None:
            raise ValueError(
                f"No overlapping groups between G and C, make sure C.obs['{self.group_id}'] holds the group identifiers of G."
            )
        raise ValueError(
            f"No overlapping groups across the levels, make sure C.obs['{self.subgroup_id}'] holds the subgroup identifiers "
            f"of S and S['{self.group_id}'] the group identifiers of G."
        )

    def update(self) -> None:
        """Synchronize the levels again after editing identifier columns in place.

        Construction and the setters keep the levels aligned. Changes made directly to
        ``C.obs[group_id]``, ``C.obs[subgroup_id]`` or the group column of the subgroup table bypass them,
        and this method applies the same matching once more.
        """
        self._match(self._G, self._S, self._C, self._G_is_table, self._S_is_table)

    def _derive(self, G, S, C, var_dims_to_sync=None) -> HiData:
        new = HiData.__new__(HiData)
        new.group_id = self.group_id
        new.subgroup_id = self.subgroup_id
        new._var_dims_to_sync = list(self._var_dims_to_sync if var_dims_to_sync is None else var_dims_to_sync)
        new.uns = _copy.deepcopy(self.uns)
        new._match(G, S, C, self._G_is_table, self._S_is_table, checked=True)
        # unchanged levels get own views
        with _keep_categories(not new._G_is_table):
            if new._G is self._G:
                new._G = new._G.copy() if new._G_is_table else new._G[:]
        if new._S is not None and new._S is self._S:
            new._S = new._S.copy() if new._S_is_table else new._S[:]
        if new._C is self._C:
            new._C = new._C[:]
        return new

    def _finish(self, new: HiData, inplace: bool) -> HiData | None:
        if not inplace:
            return new
        self._G, self._S, self._C = new._G, new._S, new._C
        self._G_is_table, self._S_is_table = new._G_is_table, new._S_is_table
        self._var_dims_to_sync = new._var_dims_to_sync
        return None

    def _actualize(self, level: Level) -> None:
        attribute = {"group": "_G", "subgroup": "_S", "cell": "_C"}[level]
        container = getattr(self, attribute)
        if container.is_view:
            setattr(self, attribute, _utils.copy_container(container))

    def _actualize_G(self) -> None:
        self._actualize("group")

    def _actualize_C(self) -> None:
        self._actualize("cell")

    def copy(self) -> HiData:
        """Return a deep copy."""
        new = HiData.__new__(HiData)
        new.group_id = self.group_id
        new.subgroup_id = self.subgroup_id
        new._var_dims_to_sync = list(self._var_dims_to_sync)
        new._G_is_table, new._S_is_table = self._G_is_table, self._S_is_table
        new._G = _utils.copy_container(self._G)
        new._S = None if self._S is None else self._S.copy()
        new._C = _utils.copy_container(self._C)
        new.uns = _copy.deepcopy(self.uns)
        return new

    @property
    def G(self) -> pd.DataFrame | AnnData | MuData:
        """Group-level data. A DataFrame when ``G`` was given as a group table."""
        return self._G.obs if self._G_is_table else self._G

    @G.setter
    def G(self, value: pd.DataFrame | AnnData | MuData) -> None:
        G, is_table = self._prepare_level(value, self.group_id)
        previous = self._G
        self._match(G, self._S, self._C, is_table, self._S_is_table)
        if is_table:
            _carry_over_matrices(previous, self._G)

    @property
    def S(self) -> pd.DataFrame | AnnData | None:
        """Subgroup-level data, a DataFrame when given as a table, or ``None`` without a subgroup level."""
        if self._S is None:
            return None
        return self._S.obs if self._S_is_table else self._S

    @S.setter
    def S(self, value: pd.DataFrame | AnnData) -> None:
        if self.subgroup_id is None:
            raise ValueError("This HiData has no subgroup level, use nest() to add one.")
        S, is_table = self._prepare_level(value, self.subgroup_id)
        if self.group_id not in S.obs.columns:
            raise ValueError(f"'{self.group_id}' not found in the subgroup table.")
        previous = self._S
        self._match(self._G, S, self._C, self._G_is_table, is_table)
        if is_table:
            _carry_over_matrices(previous, self._S)

    @property
    def C(self) -> AnnData | MuData:
        """Cell-level data, grouped by subgroup or group in the row order of the level above."""
        return self._C

    @C.setter
    def C(self, value: AnnData | MuData) -> None:
        if not isinstance(value, AnnData | MuData):
            raise TypeError(f"C must be an AnnData or MuData object, got {type(value).__name__}.")
        key = self.subgroup_id or self.group_id
        if key not in value.obs.columns:
            raise ValueError(f"'{key}' not found in C.obs")
        self._match(self._G, self._S, value, self._G_is_table, self._S_is_table)

    @property
    def G_type(self) -> Literal["table", "anndata", "mudata"]:
        """Kind of group-level data held in ``G``."""
        if self._G_is_table:
            return "table"
        return "mudata" if isinstance(self._G, MuData) else "anndata"

    @property
    def level_names(self) -> tuple[str, ...]:
        """Levels of this object from the coarsest to the cells."""
        return ("group", "subgroup", "cell") if self._S is not None else ("group", "cell")

    @property
    def levels(self) -> dict[str, AnnData | MuData]:
        """Every level as an AnnData-like object with ``obs``, ``obsm`` and ``obsp``.

        Table levels are shown through the AnnData that holds them, so group- and subgroup-aligned
        matrices of any level are reached in the same way, for example
        ``hd.levels["subgroup"].obsm["X_pca"]``. Tools that expect an AnnData with one row per unit, such
        as a sample-level AnnData, can take ``hd.levels["group"]`` directly.
        """
        result = {"group": self._G}
        if self._S is not None:
            result["subgroup"] = self._S
        result["cell"] = self._C
        return result

    def _check_level(self, level: str, allow_cell: bool = False) -> None:
        allowed = [name for name in self.level_names if allow_cell or name != "cell"]
        if level not in allowed:
            raise ValueError(f"level must be one of {allowed}, got {level!r}.")

    def _codes(self, level: str) -> np.ndarray:
        self._check_level(level)
        return self.group_codes if level == "group" else self.subgroup_codes

    def _table(self, level: str) -> pd.DataFrame:
        """Table of a level, with the group columns added to the subgroup table."""
        self._check_level(level)
        if level == "group":
            return self._G.obs
        extra = [c for c in self._G.obs.columns if c not in self._S.obs.columns]
        if not extra:
            return self._S.obs
        group_columns = _utils.broadcast(self._G.obs[extra], self.subgroup_group_codes, self._S.obs_names)
        return pd.concat([self._S.obs, group_columns], axis=1)

    @property
    def obs(self) -> pd.DataFrame:
        """Group table, one row per group. Equal to ``G`` for a table and to ``G.obs`` otherwise."""
        return self._G.obs

    @property
    def obs_names(self) -> pd.Index:
        """Group identifiers in their current order."""
        return self._G.obs_names

    @property
    def obsm(self):
        """Group-aligned matrices such as embeddings or pseudobulk profiles."""
        return self._G.obsm

    @obsm.setter
    def obsm(self, value) -> None:
        self._actualize("group")
        self._G.obsm = value

    @property
    def obsp(self):
        """Group-by-group matrices such as distances, similarities or kinship."""
        return self._G.obsp

    @obsp.setter
    def obsp(self, value) -> None:
        self._actualize("group")
        self._G.obsp = value

    @property
    def n_groups(self) -> int:
        return self._G.n_obs

    @property
    def n_subgroups(self) -> int | None:
        """Number of subgroups, or ``None`` without a subgroup level."""
        return None if self._S is None else self._S.n_obs

    @property
    def n_cells(self) -> int:
        return self._C.n_obs

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(n_groups, n_group_vars, n_cells, n_cell_vars)``. A group table has no group variables."""
        return *self._G.shape[:2], *self._C.shape

    @property
    def subgroup_codes(self) -> np.ndarray:
        """Position of the subgroup of every cell in the subgroup table."""
        if self._S is None:
            raise ValueError("This HiData has no subgroup level.")
        return _utils.codes_for(self._C.obs[self.subgroup_id], self._S.obs_names)

    @property
    def subgroup_group_codes(self) -> np.ndarray:
        """Position of the group of every subgroup in :attr:`obs_names`."""
        if self._S is None:
            raise ValueError("This HiData has no subgroup level.")
        return _utils.codes_for(self._S.obs[self.group_id], self.obs_names)

    @property
    def group_codes(self) -> np.ndarray:
        """Position in :attr:`obs_names` of the group of every cell."""
        if self._S is None:
            return _utils.codes_for(self._C.obs[self.group_id], self.obs_names)
        return self.subgroup_group_codes[self.subgroup_codes]

    @property
    def cell_offsets(self) -> np.ndarray:
        """Boundaries of the contiguous block of cells of every group in ``C``.

        The cells of group ``i`` are ``C[cell_offsets[i]:cell_offsets[i + 1]]``.
        """
        return self._offsets("group")

    def _offsets(self, level: str) -> np.ndarray:
        n = self.n_groups if level == "group" else self.n_subgroups
        return np.concatenate([[0], np.cumsum(np.bincount(self._codes(level), minlength=n))])

    @property
    def n_cells_per_group(self) -> pd.Series:
        """Number of cells of every group."""
        return pd.Series(np.diff(self._offsets("group")), index=self.obs_names, name="n_cells")

    @property
    def n_cells_per_subgroup(self) -> pd.Series:
        """Number of cells of every subgroup."""
        return pd.Series(np.diff(self._offsets("subgroup")), index=self._S.obs_names, name="n_cells")

    @property
    def n_subgroups_per_group(self) -> pd.Series:
        """Number of subgroups of every group."""
        counts = np.bincount(self.subgroup_group_codes, minlength=self.n_groups)
        return pd.Series(counts, index=self.obs_names, name="n_subgroups")

    def cells_of(self, name: str, *, level: Level = "group") -> AnnData | MuData:
        """Cells of a single group or subgroup as a view of ``C``."""
        names = self.levels[level].obs_names
        position = names.get_loc(str(name))
        offsets = self._offsets(level)
        return self._C[offsets[position] : offsets[position + 1]]

    def _sync_var_dims(self, G, C):
        kept = []
        for key in self._var_dims_to_sync:
            if key not in G.obsm:
                continue
            kept.append(key)
            frame = G.obsm[key]
            if not isinstance(frame, pd.DataFrame):
                continue
            columns = C.var_names[C.var_names.isin(frame.columns)]
            if not frame.columns.equals(columns):
                if G is self._G or G.is_view:
                    G = _detached(G)
                G.obsm[key] = G.obsm[key][columns]
        return G, kept

    def _subset(
        self, G_obs=slice(None), G_var=slice(None), C_obs=slice(None), C_var=slice(None), S_obs=slice(None)
    ) -> HiData:
        G, S, C = self._G, self._S, self._C
        if not _utils.is_full_slice(G_var) and self._G_is_table:
            raise IndexError("G is a group table and has no variables to select.")
        with _keep_categories(not self._G_is_table):
            if not _utils.is_full_slice(G_obs):
                G = G[_utils.by_label(G_obs, G.obs_names, "groups")]
            if not _utils.is_full_slice(G_var):
                G = G[:, _utils.by_label(G_var, G.var_names, "group variables")]
        if not _utils.is_full_slice(S_obs):
            if S is None:
                raise IndexError("This HiData has no subgroup level.")
            S = S[_utils.by_label(S_obs, S.obs_names, "subgroups")]
        if not _utils.is_full_slice(C_obs):
            C = C[_utils.by_label(C_obs, C.obs_names, "cells")]
        if not _utils.is_full_slice(C_var):
            C = C[:, _utils.by_label(C_var, C.var_names, "cell variables")]
        G, kept = self._sync_var_dims(G, C)
        return self._derive(G, S, C, kept)

    def sel(
        self,
        *,
        G_obs=slice(None),
        G_var=slice(None),
        C_obs=slice(None),
        C_var=slice(None),
        S_obs=slice(None),
    ) -> HiData:
        """Select along the axes of every level by keyword.

        A boolean ``pandas.Series`` is matched to the axis by name, as in ``filter_groups``, ``filter_subgroups`` and
        ``filter_cells``, and an entry that is missing (NA) counts as ``False``. A Series with the default index
        0, 1, 2, ... raises an error, because it cannot be matched by name. Boolean arrays and lists are applied by
        position, in the order of this object.

        Parameters
        ----------
        G_obs
            Groups to keep.
        G_var
            Group variables to keep. Not available for a group table.
        C_obs
            Cells to keep.
        C_var
            Cell variables to keep.
        S_obs
            Subgroups to keep.

        Returns
        -------
        HiData
            Units that lose their connection to the other levels are dropped.
        """
        return self._subset(G_obs, G_var, C_obs, C_var, S_obs)

    def __getitem__(self, key) -> HiData:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 4:
            raise IndexError("HiData only supports 4 dimensions")
        key = self._expand_ellipsis(key)
        key = tuple((index,) if isinstance(index, str) else index for index in key)
        key = key + (slice(None),) * (4 - len(key))
        return self._subset(*key)

    def _expand_ellipsis(self, index_tuple):
        n_ellipsis = sum(1 for index in index_tuple if index is Ellipsis)
        if n_ellipsis == 0:
            return index_tuple
        if n_ellipsis > 1:
            raise IndexError("HiData only supports a single ellipsis ('...').")
        position = next(i for i, index in enumerate(index_tuple) if index is Ellipsis)  # ``==`` fails on arrays
        n_full = 4 - (len(index_tuple) - 1)
        return index_tuple[:position] + (slice(None),) * n_full + index_tuple[position + 1 :]

    def filter_groups(self, key: Selection, *, inplace: bool = False) -> HiData | None:
        """Keep a subset of groups together with their subgroups and cells.

        Parameters
        ----------
        key
            A query on the group table such as ``"age > 60 and sex == 'F'"``, a callable that takes
            the group table and returns a boolean mask, a boolean mask over groups, or group
            identifiers.
            A boolean ``pandas.Series`` is matched by name and its missing entries (NA) count as
            ``False``, so a unit without an answer is not kept. A boolean array or list is applied by position.
            Names are given as text and must exist. Integers and slices raise an error; select by
            position with ``hd[...]`` or :meth:`sel`.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self.obs, "groups")
        return self._finish(self._subset(G_obs=mask), inplace)

    def filter_subgroups(self, key: Selection, *, inplace: bool = False) -> HiData | None:
        """Keep a subset of subgroups together with their cells. Groups without subgroups are dropped.

        Parameters
        ----------
        key
            A query on the subgroup table, which also sees group columns, such as
            ``"timepoint == 'day 7' and age > 60"``, a callable that takes this table and returns a
            boolean mask, a boolean mask over subgroups, or subgroup identifiers.
            A boolean ``pandas.Series`` is matched by name and its missing entries (NA) count as
            ``False``, so a unit without an answer is not kept. A boolean array or list is applied by position.
            Names are given as text and must exist. Integers and slices raise an error; select by
            position with ``hd[...]`` or :meth:`sel`.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self._table("subgroup"), "subgroups")
        return self._finish(self._subset(S_obs=mask), inplace)

    def filter_cells(self, key: Selection, *, inplace: bool = False) -> HiData | None:
        """Keep a subset of cells. Subgroups and groups that lose all their cells are dropped.

        Parameters
        ----------
        key
            A query on ``C.obs`` such as ``"cell_type == 'B cell'"``, a callable that takes ``C.obs``
            and returns a boolean mask, a boolean mask over cells, or cell names.
            A boolean ``pandas.Series`` is matched by name and its missing entries (NA) count as
            ``False``, so a unit without an answer is not kept. A boolean array or list is applied by position.
            Names are given as text and must exist. Integers and slices raise an error; select by
            position with ``hd[...]`` or :meth:`sel`.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self._C.obs, "cells")
        return self._finish(self._subset(C_obs=mask), inplace)

    def _subset_level(self, level: str, mask) -> HiData:
        return self._subset(G_obs=mask) if level == "group" else self._subset(S_obs=mask)

    def filter_groups_by_cells(
        self,
        min_cells: int,
        *,
        groupby: str | None = None,
        groups: Sequence | None = None,
        level: Level = "group",
        inplace: bool = False,
    ) -> HiData | None:
        """Keep groups, or subgroups, with enough cells, optionally in every one of several cell groups.

        Pseudobulk and composition analyses need each unit to contribute enough cells of every cell
        type that is compared. With ``groupby`` a unit is kept only if it has at least ``min_cells``
        cells in each of ``groups``.

        Parameters
        ----------
        min_cells
            Minimum number of cells.
        groupby
            Column of ``C.obs`` defining cell groups, for example cell types.
        groups
            Cell groups that must reach ``min_cells``. Defaults to every cell group of ``groupby`` except
            ``"nan"``, the cells whose value is missing.
        level
            ``"group"`` or ``"subgroup"``.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        if groupby is None:
            keep = np.diff(self._offsets(level)) >= min_cells
        else:
            counts = self._cell_group_counts(groupby, level)
            groups = [g for g in counts.columns if g != MISSING_GROUP] if groups is None else list(groups)
            missing = [g for g in groups if g not in counts.columns]
            if missing:
                raise KeyError(f"Cell groups {missing} not found in C.obs['{groupby}'].")
            keep = (counts[groups] >= min_cells).all(axis=1).to_numpy()
        return self._finish(self._subset_level(level, keep), inplace)

    def subsample_cells(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        groupby: str | None = None,
        level: Level = "group",
        random_state: int | None = 0,
        inplace: bool = False,
    ) -> HiData | None:
        """Randomly subsample cells within every group or subgroup.

        Caps how much any single unit contributes, for example to stop groups with many cells from
        dominating a model, while keeping every unit.

        Parameters
        ----------
        n
            Maximum number of cells to keep per unit, or per unit and cell group with ``groupby``.
            Smaller units keep all their cells.
        fraction
            Fraction of cells to keep per unit instead of a fixed number. At least one cell is kept.
        groupby
            Column of ``C.obs`` to stratify by within units, for example cell types. Cells whose value is
            missing, or a string such as ``"nan"`` or ``"NA"``, form a cell group of their own.
        level
            ``"group"`` or ``"subgroup"``.
        random_state
            Seed of the random generator.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The subsampled object, or ``None`` when ``inplace`` is set.
        """
        if (n is None) == (fraction is None):
            raise ValueError("Provide exactly one of n or fraction.")
        strata = self._codes(level)
        if groupby is not None:
            category_codes, categories = pd.factorize(_utils.missing_as_na(self._C.obs[groupby]), use_na_sentinel=False)
            strata = strata * len(categories) + category_codes
        strata_codes, strata = pd.factorize(strata)
        sizes = np.bincount(strata_codes)
        if n is not None:
            limits = np.minimum(sizes, n)
        else:
            limits = np.maximum(np.round(sizes * fraction), 1).astype(int)
        rng = np.random.default_rng(random_state)
        order = np.lexsort((rng.random(strata_codes.size), strata_codes))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size) - starts[strata_codes[order]]
        keep = rank < limits[strata_codes]
        return self._finish(self._subset(C_obs=keep), inplace)

    def complete_cases(
        self, mods: Sequence[str] | None = None, *, min_cells: int = 1, level: Level = "group", inplace: bool = False
    ) -> HiData | None:
        """Keep groups, or subgroups, that have cells in every modality of a MuData ``C``.

        Parameters
        ----------
        mods
            Modalities that must be present. Defaults to all of them.
        min_cells
            Minimum number of cells per modality.
        level
            ``"group"`` or ``"subgroup"``.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        HiData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        coverage = self.coverage(level=level)
        mods = list(coverage.columns) if mods is None else list(mods)
        keep = (coverage[mods] >= min_cells).all(axis=1).to_numpy()
        return self._finish(self._subset_level(level, keep), inplace)

    def coverage(self, *, level: Level = "group") -> pd.DataFrame:
        """Number of cells of every group, or subgroup, measured in each modality of ``C``.

        A MuData can pair assays measured on different cells of the same groups, for example blood
        scRNA-seq with tissue scATAC-seq. This table shows which units have which assays.

        Parameters
        ----------
        level
            ``"group"`` or ``"subgroup"``.

        Returns
        -------
        pandas.DataFrame
            Units by modalities, or a single column ``"C"`` when ``C`` is an AnnData.
        """
        codes = self._codes(level)
        names = self.levels[level].obs_names
        if isinstance(self._C, AnnData):
            return pd.DataFrame({"C": np.bincount(codes, minlength=len(names))}, index=names)
        counts = {}
        for name in self._C.mod:
            present = np.asarray(self._C.obsmap[name]).ravel() > 0
            counts[name] = np.bincount(codes[present], minlength=len(names))
        return pd.DataFrame(counts, index=names)

    def _cell_group_counts(self, groupby: str, level: str = "group") -> pd.DataFrame:
        codes = self._codes(level)
        names = self.levels[level].obs_names
        if groupby in self._C.obsm:
            membership = self._C.obsm[groupby]
            labels = membership.columns if isinstance(membership, pd.DataFrame) else pd.RangeIndex(membership.shape[1])
            counts = _utils.group_sum(membership, codes, len(names))
            return pd.DataFrame(counts, index=names, columns=pd.Index(labels.astype(str), name=groupby))
        category_codes, categories = _category_codes(self._C.obs[groupby])
        flat = codes * len(categories) + category_codes
        counts = np.bincount(flat, minlength=len(names) * len(categories)).reshape(len(names), len(categories))
        return pd.DataFrame(counts, index=names, columns=pd.Index(categories, name=groupby))

    def composition(
        self,
        groupby: str,
        *,
        normalize: bool = True,
        clr: bool = False,
        pseudocount: float = 0.5,
        level: Level = "group",
        key_added: str | None = None,
    ) -> pd.DataFrame:
        """Cell group composition of every group or subgroup.

        Parameters
        ----------
        groupby
            Column of ``C.obs`` with cell groups, for example cell types, or a key of ``C.obsm`` with
            a cells by cell groups membership matrix. A membership matrix allows overlapping cell groups,
            such as the neighbourhoods of Milo, and yields the units by neighbourhoods count matrix.
            Cells whose value in the column is missing, or a string such as ``"nan"`` or ``"NA"``, form
            the cell group ``"nan"``.
        normalize
            Return fractions that sum to one per unit instead of counts.
        clr
            Return the centered log-ratio of the counts plus ``pseudocount``, the usual transform for
            comparing compositions.
        pseudocount
            Added to all counts before the centered log-ratio.
        level
            ``"group"`` or ``"subgroup"``.
        key_added
            Key of the level's ``obsm`` to store the result under. Defaults to
            ``"{groupby}_clr"``, ``"{groupby}_fractions"`` or ``"{groupby}_counts"``.

        Returns
        -------
        pandas.DataFrame
            Units by cell groups. The same frame is stored in ``obsm`` of the level as a snapshot that
            follows selections but is not recomputed when cells are filtered.
        """
        counts = self._cell_group_counts(groupby, level)
        if clr:
            logs = np.log(counts.to_numpy(dtype=np.float64) + pseudocount)
            result = pd.DataFrame(logs - logs.mean(axis=1, keepdims=True), index=counts.index, columns=counts.columns)
            suffix = "clr"
        elif normalize:
            result = counts.astype(np.float64).div(counts.sum(axis=1), axis=0)
            suffix = "fractions"
        else:
            result = counts.astype(np.int64)
            suffix = "counts"
        result.columns = result.columns.astype(str)
        self._actualize(level)
        self.levels[level].obsm[key_added or f"{groupby}_{suffix}"] = result
        return result

    def infer_levels(self) -> pd.DataFrame:
        """Find the level every annotation naturally belongs to.

        A column of ``C.obs`` that is constant within every group is group-level information that is
        repeated on cells, and one that is constant within subgroups is subgroup-level. The same holds
        for columns of the subgroup table.

        Returns
        -------
        pandas.DataFrame
            One row per column with the table it is stored in and its ``level``.
        """
        rows = []
        group_codes = self.group_codes
        group_first = _utils.first_positions(group_codes, self.n_groups)
        if self._S is not None:
            subgroup_codes = self.subgroup_codes
            subgroup_first = _utils.first_positions(subgroup_codes, self.n_subgroups)
            subgroup_groups = self.subgroup_group_codes
            subgroup_group_first = _utils.first_positions(subgroup_groups, self.n_groups)
            for column in self._S.obs.columns:
                if column in (self.group_id, self.subgroup_id):
                    continue
                constant = _utils.is_constant_within(self._S.obs[column], subgroup_groups, subgroup_group_first)
                rows.append((column, "subgroup", "group" if constant else "subgroup"))
        for column in self._C.obs.columns:
            if column in (self.group_id, self.subgroup_id):
                continue
            values = self._C.obs[column]
            if _utils.is_constant_within(values, group_codes, group_first):
                level = "group"
            elif self._S is not None and _utils.is_constant_within(values, subgroup_codes, subgroup_first):
                level = "subgroup"
            else:
                level = "cell"
            rows.append((column, "cell", level))
        return pd.DataFrame(rows, columns=["column", "table", "level"]).set_index("column")

    def get_df(
        self,
        keys: str | Sequence[str],
        *,
        level: Level = "cell",
        func: str | Callable = "mean",
        layer: str | None = None,
        mod: str | None = None,
    ) -> pd.DataFrame:
        """Collect variables from any level into one table at the requested level.

        Keys can name group or subgroup columns, cell annotations, genes or other variables of ``C``,
        variables of an AnnData ``G``, or ``obsm`` keys of any level, which expand into one column per
        dimension. Values from coarser levels are repeated for their members, values from finer
        levels are aggregated with ``func`` when numeric and kept when constant within the unit. A
        prefix such as ``"G:age"``, ``"S:timepoint"`` or ``"C:n_genes"`` selects the level when a name
        exists at several levels, and ``"rna:CD14"`` selects a modality of a MuData ``C``.

        Parameters
        ----------
        keys
            Names of the variables to collect.
        level
            ``"cell"``, ``"subgroup"`` or ``"group"``.
        func
            Aggregation of numeric values moved from finer levels.
        layer
            Layer of ``C`` to read genes from instead of ``X``.
        mod
            Modality of a MuData ``C`` to read genes from.

        Returns
        -------
        pandas.DataFrame
            Indexed by the units of ``level``.

        Examples
        --------
        >>> hd.get_df(["age", "severity", "cell_type"], level="cell")
        >>> hd.get_df(["age", "CD14", "X_scVI"], level="group")
        """
        from hidata._core._getter import get_df

        return get_df(self, keys, level=level, func=func, layer=layer, mod=mod)

    def push_obs(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        level: Level = "group",
        to_level: Level = "cell",
        overwrite: bool = False,
    ) -> None:
        """Broadcast columns of a level table to the members of every unit.

        Parameters
        ----------
        columns
            Columns of the table of ``level``. Defaults to all of them except the identifiers.
        level
            Level the columns come from, ``"group"`` or ``"subgroup"``.
        to_level
            Finer level to write them to, ``"cell"`` or ``"subgroup"``.
        overwrite
            Replace columns that already exist at ``to_level``.
        """
        self._check_level(level)
        self._check_level(to_level, allow_cell=True)
        if _RANK[to_level] <= _RANK[level]:
            raise ValueError("to_level must be finer than level.")
        source = self.levels[level].obs
        columns = self._level_columns(source, columns)
        target = self.levels[to_level].obs
        clashes = [c for c in columns if c in target.columns]
        if clashes and not overwrite:
            raise ValueError(f"Columns {clashes} already exist at the {to_level} level, pass overwrite=True.")
        codes = self.subgroup_group_codes if to_level == "subgroup" else self._codes(level)
        self._actualize(to_level)
        target = self.levels[to_level].obs
        for column in columns:
            target[column] = source[column].array.take(codes)

    def pull_obs(
        self,
        columns: str | Sequence[str] | Mapping[str, str] | None = None,
        *,
        func: str | Callable | None = None,
        level: Level = "group",
        from_level: Level = "cell",
        drop: bool = False,
        overwrite: bool = False,
    ) -> list[str]:
        """Collect columns of a finer level into the table of a coarser one.

        Parameters
        ----------
        columns
            Columns to collect, or a mapping from their names to the names they get in the table of
            ``level``. By default every column that is constant within the units and not yet in the
            target table.
        func
            Aggregation for columns that vary within units, for example ``"mean"`` or ``"median"``.
            Without it such columns raise an error.
        level
            Level to collect into, ``"group"`` or ``"subgroup"``.
        from_level
            Finer level to collect from, ``"cell"`` or ``"subgroup"``.
        drop
            Remove the collected columns from the finer level.
        overwrite
            Replace columns that already exist in the target table.

        Returns
        -------
        list of str
            The names of the collected columns in the target table.

        Examples
        --------
        >>> hd.pull_obs({"n_genes": "median_n_genes", "pct_mito": "median_pct_mito"}, func="median")
        """
        self._check_level(level)
        self._check_level(from_level, allow_cell=True)
        if _RANK[from_level] <= _RANK[level]:
            raise ValueError("from_level must be finer than level.")
        source = self.levels[from_level].obs
        target = self.levels[level].obs
        codes = self.subgroup_group_codes if from_level == "subgroup" else self._codes(level)
        n_units = len(target)
        first = _utils.first_positions(codes, n_units)
        if columns is None:
            exclude = {self.group_id, self.subgroup_id} | (set() if overwrite else set(target.columns))
            columns = _utils.constant_columns(source, codes, first, exclude=exclude)
        if isinstance(columns, Mapping):
            names = dict(columns)
        else:
            names = {c: c for c in ([columns] if isinstance(columns, str) else columns)}
        clashes = [c for c in names.values() if c in target.columns]
        if clashes and not overwrite:
            raise ValueError(f"Columns {clashes} already exist in the {level} table, pass overwrite=True.")

        values = {}
        for column, name in names.items():
            series = source[column]
            is_numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
            if func is not None and is_numeric:
                collected = series.groupby(codes, observed=True).agg(func)
                collected = collected.reindex(np.arange(n_units))
                collected.index = target.index
            elif _utils.is_constant_within(series, codes, first):
                collected = series.iloc[first]
                collected.index = target.index
            elif func is None:
                raise ValueError(f"'{column}' varies within {level}s, pass func= to aggregate it.")
            else:
                collected = series.groupby(codes, observed=True).agg(func)
                collected = collected.reindex(np.arange(n_units))
                collected.index = target.index
            values[name] = collected

        self._actualize(level)
        target = self.levels[level].obs
        for name, collected in values.items():
            target[name] = collected
        if drop and names:
            self._actualize(from_level)
            container = self.levels[from_level]
            container.obs = container.obs.drop(columns=list(names))
        return list(values)

    def push_obsm(self, key: str, *, level: Level = "group", key_added: str | None = None) -> None:
        """Broadcast a group- or subgroup-level matrix to every cell in ``C.obsm``.

        Parameters
        ----------
        key
            Key of the level's ``obsm``, for example a patient embedding.
        level
            ``"group"`` or ``"subgroup"``.
        key_added
            Key in ``C.obsm``. Defaults to ``key``.
        """
        matrix = self.levels[level].obsm[key]
        codes = self._codes(level)
        if isinstance(matrix, pd.DataFrame):
            broadcast = matrix.iloc[codes]
            broadcast.index = self._C.obs_names
        else:
            broadcast = matrix[codes]
        self._actualize("cell")
        self._C.obsm[key_added or key] = broadcast

    def to_anndata(
        self, columns: str | Sequence[str] | None = None, *, obsm: Sequence[str] = (), level: Level = "group"
    ) -> AnnData | MuData:
        """Cell-level object with group and subgroup annotations attached to every cell.

        Useful to hand the data to tools that only understand a single AnnData, for example to color
        cells by patient covariates. Matrices are shared with ``C``, not copied.

        Parameters
        ----------
        columns
            Columns of the group table, and of the subgroup table when there is one, to attach.
            Defaults to all of them except identifiers already present in ``C.obs``.
        obsm
            Keys of the ``obsm`` of ``level`` to attach to ``obsm`` of the result.
        level
            Level whose ``obsm`` the keys in ``obsm`` refer to.

        Returns
        -------
        AnnData or MuData
            Of the same type as ``C``.
        """
        table = self._table("subgroup" if self._S is not None else "group")
        codes = self.subgroup_codes if self._S is not None else self.group_codes
        if columns is None:
            columns = [c for c in table.columns if c not in self._C.obs.columns]
        else:
            columns = self._level_columns(table, columns)
        clashes = [c for c in columns if c in self._C.obs.columns]
        if clashes:
            raise ValueError(f"Columns {clashes} exist both at a unit level and in C.obs.")
        obs = self._C.obs.assign(**{c: table[c].array.take(codes) for c in columns})
        result = _utils.shallow_with_obs(self._C, obs)
        unit_codes = self._codes(level)
        for key in obsm:
            matrix = self.levels[level].obsm[key]
            if isinstance(matrix, pd.DataFrame):
                matrix = matrix.iloc[unit_codes].set_axis(self._C.obs_names)
            else:
                matrix = matrix[unit_codes]
            result.obsm[key] = matrix
        return result

    def _level_columns(self, table: pd.DataFrame, columns) -> list[str]:
        if columns is None:
            return [c for c in table.columns if c not in (self.group_id, self.subgroup_id)]
        columns = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise KeyError(f"Columns {missing} not found.")
        return columns

    def cell_weights(self, *, level: Level = "group", by: str | None = None) -> pd.Series:
        """Weights that let every group, or subgroup, contribute equally to a cell-level model.

        Each cell gets the inverse number of cells of its unit, so groups with many cells do not
        dominate a loss or a fit. With ``by`` the classes of a unit-level column, such as disease
        status, are balanced as well. Weights are scaled to a mean of one.

        Parameters
        ----------
        level
            ``"group"`` or ``"subgroup"``.
        by
            Column of the unit table whose classes get equal total weight.

        Returns
        -------
        pandas.Series
            One weight per cell.
        """
        codes = self._codes(level)
        sizes = np.bincount(codes)
        weights = 1.0 / sizes[codes]
        if by is not None:
            classes = pd.factorize(self._table(level)[by], use_na_sentinel=False)[0]
            units_per_class = np.bincount(classes)
            weights = weights / units_per_class[classes[codes]]
        weights *= weights.size / weights.sum()
        return pd.Series(weights, index=self._C.obs_names, name="weight")

    def aggregate(
        self,
        *,
        key_added: None | str = None,
        layer: None | str = None,
        obs: None | str | list[str] = None,
        obsm: None | str = None,
        filter_key: None | str = None,
        filter_value=None,
        add_to_obs: bool = False,
        func: str | Callable = "mean",
        sync_var: bool = False,
        verbose: bool = False,
        mod: str | None = None,
        level: Level = "group",
    ) -> None:
        """Aggregate cell-level data to groups or subgroups.

        Aggregates ``C.X`` unless one of ``layer``, ``obsm`` or ``obs`` is given. Matrices are stored
        in ``obsm`` of the level, columns of ``C.obs`` in ``obsm`` or, with ``add_to_obs``, in the
        level table.

        Parameters
        ----------
        key_added
            Key of the result. Defaults to ``"{slot}_{func}"``, followed by ``filter_value`` when
            filtering.
        layer
            Layer of ``C`` to aggregate.
        obs
            Column or columns of ``C.obs`` to aggregate.
        obsm
            Key of ``C.obsm`` to aggregate.
        filter_key
            Column of ``C.obs`` used to restrict the cells, for example a cell type column.
        filter_value
            Value of ``filter_key`` that cells must have.
        add_to_obs
            Store aggregated ``obs`` columns in the level table instead of ``obsm``.
        func
            Aggregation for matrices, one of ``"sum"``, ``"mean"``, ``"var"``, ``"count_nonzero"``,
            ``"median"``, ``"min"`` or ``"max"``. For ``obs`` any pandas aggregation is accepted.
        sync_var
            Keep the columns of the result in sync with ``C.var_names`` when genes are selected.
        verbose
            Log what was aggregated.
        mod
            Modality to aggregate when ``C`` is a MuData. Modalities measured on a part of the cells
            are aggregated over those cells.
        level
            ``"group"`` or ``"subgroup"``.
        """
        if filter_key is not None and filter_value is None:
            raise ValueError("filter_value must be provided if filter_key is provided")
        if sum(x is not None for x in (layer, obsm, obs)) > 1:
            raise ValueError("Only one of layer, obsm or obs can be provided")

        codes = self._codes(level)
        container = self.levels[level]
        n_units, names = container.n_obs, container.obs_names
        if filter_key is not None:
            codes = np.where((self._C.obs[filter_key] == filter_value).to_numpy(), codes, -1)

        if obs is not None:
            slot = obs if isinstance(obs, str) else "_".join(obs)
        else:
            slot = layer or obsm or "X"
        if key_added is None:
            key_added = f"{slot}_{func if isinstance(func, str) else func.__name__}"
            if filter_value is not None:
                key_added += f"_{filter_value}"

        if obs is not None:
            columns = [obs] if isinstance(obs, str) else list(obs)
            member = codes >= 0
            result = self._C.obs.loc[member, columns].groupby(codes[member], observed=True).agg(func)
            result = result.reindex(np.arange(n_units))
            result.index = names
            self._actualize(level)
            container = self.levels[level]
            if add_to_obs:
                for column in columns:
                    container.obs[column] = result[column]
            else:
                container.obsm[key_added] = result
        else:
            if not isinstance(func, str):
                raise TypeError("func must be a string when aggregating matrices.")
            matrix, rows = _utils.cell_matrix(self._C, layer=layer, obsm=obsm, mod=mod)
            values = _utils.aggregate_rows(matrix, codes if rows is None else codes[rows], n_units, func)
            if obsm is not None:
                result = (
                    pd.DataFrame(values, index=names, columns=matrix.columns)
                    if isinstance(matrix, pd.DataFrame)
                    else values
                )
            else:
                var_names = _utils.modality(self._C, mod)[0].var_names
                result = pd.DataFrame(values, index=names, columns=var_names)
                if sync_var and level == "group" and key_added not in self._var_dims_to_sync:
                    self._var_dims_to_sync.append(key_added)
            self._actualize(level)
            self.levels[level].obsm[key_added] = result

        if verbose:
            n_observed = np.unique(codes[codes >= 0]).size
            logger.info("Aggregated %s to %s, observed for %d %ss.", slot, key_added, n_observed, level)

    def pseudobulk(
        self,
        *,
        layer: str | None = None,
        groupby: str | None = None,
        func: str = "sum",
        min_cells: int = 1,
        mod: str | None = None,
        level: Level = "group",
    ) -> AnnData:
        """Aggregate cells into one profile per group or subgroup, or per unit and cell group.

        The result carries the unit table in ``obs``, with group columns added for subgroups, and is
        ready for differential testing with covariates of both levels in the design. ``n_cells`` and
        ``total_counts`` record how many cells and counts every profile is built from. Without
        ``groupby`` it also carries ``obsm``, ``obsp`` and :attr:`uns` of the level, so it is a
        complete unit-level AnnData that can be assigned to ``G`` or used with tools that expect one
        AnnData per cohort.

        Parameters
        ----------
        layer
            Layer of ``C`` to aggregate, ``X`` by default. Raw counts suit differential testing.
        groupby
            Column of ``C.obs``, for example cell types, giving one profile per unit and cell group.
            Cells whose value is missing, or a string such as ``"nan"`` or ``"NA"``, form the cell group
            ``"nan"``.
        func
            Aggregation, see :meth:`aggregate`.
        min_cells
            Profiles built from fewer cells are dropped.
        mod
            Modality to aggregate when ``C`` is a MuData.
        level
            ``"group"`` or ``"subgroup"``.

        Returns
        -------
        anndata.AnnData
            Profiles by genes, with ``obs`` holding the unit table, ``groupby``, ``n_cells`` and
            ``total_counts``.
        """
        adata, rows = _utils.modality(self._C, mod)
        matrix, _ = _utils.cell_matrix(self._C, layer=layer, mod=mod)
        container = self.levels[level]
        names = container.obs_names
        unit_codes = self._codes(level)
        if rows is not None:
            unit_codes = unit_codes[rows]
        if groupby is None:
            codes, n_codes, categories = unit_codes, len(names), None
        else:
            labels = self._C.obs[groupby]
            if rows is not None:
                labels = labels.iloc[rows]
            category_codes, categories = _category_codes(labels)
            codes = unit_codes * len(categories) + category_codes
            n_codes = len(names) * len(categories)

        counts = np.bincount(codes[codes >= 0], minlength=n_codes)
        keep = np.flatnonzero(counts >= max(min_cells, 1))
        values = _utils.aggregate_rows(matrix, codes, n_codes, func)[keep]
        sums = values if func == "sum" else _utils.aggregate_rows(matrix, codes, n_codes, "sum")[keep]
        totals = sums.sum(axis=1)

        unit_positions = keep if groupby is None else keep // len(categories)
        obs = self._table(level).iloc[unit_positions].copy()
        key = self.group_id if level == "group" else self.subgroup_id
        if key not in obs.columns:
            obs.insert(0, key, obs.index.to_numpy())
        if groupby is None:
            obs.index = names[unit_positions]
        else:
            category_values = categories[keep % len(categories)]
            obs[groupby] = pd.Categorical(category_values, categories=categories)
            obs.index = [f"{d}_{g}" for d, g in zip(names[unit_positions], category_values, strict=True)]
        obs.index.name = None
        obs["n_cells"] = counts[keep]
        obs["total_counts"] = totals
        if groupby is not None:
            return AnnData(X=values, obs=obs, var=adata.var.copy())
        with _keep_categories(level == "group" and not self._G_is_table):
            units = container[keep]
        return AnnData(
            X=values,
            obs=obs,
            var=adata.var.copy(),
            obsm=dict(units.obsm),
            obsp=dict(units.obsp),
            uns=_copy.deepcopy(self.uns),
        )

    def collapse(
        self,
        by: str,
        *,
        func: str | dict | None = None,
        obsm_func: str | None = "mean",
        count_key: str | None = None,
    ) -> HiData:
        """Merge groups into coarser units given by a group table column.

        Moves one level up, for example from guides to their target genes, from wells to drugs or
        from donors to sites. Cells keep their original group in ``C.obs`` and gain the new unit in
        ``C.obs[by]``. A subgroup level is kept, with every subgroup assigned to its new unit. Use
        :meth:`nest` instead to keep the merged groups as a subgroup level.

        Parameters
        ----------
        by
            Group table column naming the new unit of every group. Groups whose value is missing, or a string
            such as ``"nan"`` or ``"NA"``, are dropped with a warning, as for group and subgroup identifiers.
        func
            Aggregation for group table columns that vary within a unit, either one pandas
            aggregation for all numeric columns or a dict per column. Without it such columns are
            dropped.
        obsm_func
            Aggregation of :attr:`obsm` matrices over the groups of a unit, see :meth:`aggregate`.
            ``None`` drops them. Group-by-group matrices in :attr:`obsp` are always dropped.
        count_key
            Column with the number of merged groups. Defaults to ``"n_{group_id}"``.

        Returns
        -------
        HiData
            With a group table indexed by the values of ``by``.
        """
        self = self._with_units(by)
        unit_codes, table = self._units_of(by, func=func, count_key=count_key)
        obsm = {}
        if obsm_func is not None:
            for key, matrix in self.obsm.items():
                values = _utils.aggregate_rows(matrix, unit_codes, len(table), obsm_func)
                if isinstance(matrix, pd.DataFrame):
                    values = pd.DataFrame(values, index=table.index, columns=matrix.columns)
                obsm[key] = values

        cell_units = pd.Categorical(table.index[unit_codes[self.group_codes]], categories=table.index)
        cells = _utils.shallow_with_obs(self._C, self._C.obs.assign(**{by: cell_units}))[:]
        synced = [k for k in self._var_dims_to_sync if k in obsm]
        if self._S is None:
            return HiData(
                G=table, C=cells, group_id=by, obsm=obsm, uns=_copy.deepcopy(self.uns), var_dims_to_sync=synced
            )
        subgroups = _utils.copy_container(self._S)
        subgroups.obs[by] = np.asarray(table.index[unit_codes[self.subgroup_group_codes]], dtype=str)
        new = HiData(
            G=table,
            S=subgroups.obs if self._S_is_table else subgroups,
            C=cells,
            group_id=by,
            subgroup_id=self.subgroup_id,
            obsm=obsm,
            uns=_copy.deepcopy(self.uns),
            var_dims_to_sync=synced,
        )
        if self._S_is_table:
            _copy_matrices(subgroups, new._S)
        return new

    def _with_units(self, by: str) -> HiData:
        """Drop the groups whose ``by`` is missing, as the constructor does for group and subgroup identifiers."""
        if by not in self.obs.columns:
            raise KeyError(f"'{by}' not found in the group table.")
        _log_missing_labels(self.obs[by], "groups", by)
        missing = _utils.is_missing_label(self.obs[by])
        return self.filter_groups(~missing) if missing.any() else self

    def _units_of(self, by: str, *, func=None, count_key: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if by not in self.obs.columns:
            raise KeyError(f"'{by}' not found in the group table.")
        units = self.obs[by]
        unit_codes, unit_names = pd.factorize(pd.Series(_utils.string_labels(units)), sort=True)
        unit_names = pd.Index(np.asarray(unit_names, dtype=str), name=by)
        first = _utils.first_positions(unit_codes, len(unit_names))
        constant = _utils.constant_columns(self.obs, unit_codes, first, exclude=(by, self.group_id))
        table = self.obs.iloc[first][constant].copy()
        table.index = unit_names
        if func is not None:
            varying = [c for c in self.obs.columns if c not in constant and c not in (by, self.group_id)]
            if isinstance(func, dict):
                plan = {c: f for c, f in func.items() if c in varying}
            else:
                plan = {c: func for c in varying if pd.api.types.is_numeric_dtype(self.obs[c])}
            if plan:
                aggregated = self.obs[list(plan)].groupby(unit_codes).agg(plan)
                aggregated.index = unit_names
                table = table.join(aggregated)
        table[count_key or f"n_{self.group_id}"] = np.bincount(unit_codes, minlength=len(unit_names))
        return unit_codes, table

    def nest(self, by: str, *, keep_in_subgroups: bool = False) -> HiData:
        """Turn the groups into subgroups of coarser groups given by a group table column.

        Unlike :meth:`collapse` nothing is merged away: the current group table, with its matrices,
        becomes the subgroup table and the values of ``by`` become the new groups. Columns that are
        constant within the new groups move to their table. This is how a cohort built from samples,
        for example a patient on a collection day, gains its patient level.

        Parameters
        ----------
        by
            Group table column naming the new group of every current group. Groups whose value is missing, or a string
            such as ``"nan"`` or ``"NA"``, are dropped with a warning, as for group and subgroup identifiers.
        keep_in_subgroups
            Keep the moved columns in the subgroup table as well.

        Returns
        -------
        HiData
            With three levels, where ``subgroup_id`` is the current ``group_id`` and ``group_id`` is
            ``by``.
        """
        if self._S is not None:
            raise ValueError("This HiData already has a subgroup level.")
        self = self._with_units(by)
        unit_codes, table = self._units_of(by)
        table = table.drop(columns=[f"n_{self.group_id}"])
        subgroups = _utils.copy_container(self._G)
        subgroups.obs[by] = np.asarray(table.index[unit_codes], dtype=str)
        if not keep_in_subgroups:
            subgroups.obs = subgroups.obs.drop(columns=[c for c in table.columns if c in subgroups.obs.columns])
        subgroup_side = subgroups.obs if self._G_is_table else subgroups
        new = HiData(
            G=table, S=subgroup_side, C=self._C[:], group_id=by, subgroup_id=self.group_id, uns=_copy.deepcopy(self.uns)
        )
        if self._G_is_table:
            _copy_matrices(subgroups, new._S)
        return new

    def flatten(self) -> HiData:
        """Drop the group level and make the subgroups the units.

        Group columns and group-aligned matrices are repeated for every subgroup of a group, where they
        do not clash with subgroup columns or matrices. Group-by-group matrices are dropped.

        Returns
        -------
        HiData
            With two levels, where ``group_id`` is the current ``subgroup_id``.
        """
        if self._S is None:
            raise ValueError("This HiData has no subgroup level.")
        codes = self.subgroup_group_codes
        subgroups = _utils.copy_container(self._S)
        extra = [c for c in self._G.obs.columns if c not in subgroups.obs.columns]
        if extra:
            group_columns = _utils.broadcast(self._G.obs[extra], codes, subgroups.obs_names)
            subgroups.obs = pd.concat([subgroups.obs, group_columns], axis=1)
        for key, matrix in self._G.obsm.items():
            if key in subgroups.obsm:
                continue
            if isinstance(matrix, pd.DataFrame):
                subgroups.obsm[key] = matrix.iloc[codes].set_axis(subgroups.obs_names)
            else:
                subgroups.obsm[key] = matrix[codes]
        side = subgroups.obs if self._S_is_table else subgroups
        new = HiData(G=side, C=self._C[:], group_id=self.subgroup_id, uns=_copy.deepcopy(self.uns))
        if self._S_is_table:
            _copy_matrices(subgroups, new._G)
        return new

    def _split_values(self, stratify, groups) -> tuple[pd.Series | None, pd.Series | None]:
        """``stratify`` and ``groups`` as one value per group, with every missing value as NaN."""
        if isinstance(stratify, str):
            stratify = self.obs[stratify]
        elif isinstance(stratify, pd.Series):
            stratify = _utils.align_series(stratify, self.obs_names, "groups", "stratify")
        elif stratify is not None:
            if len(stratify) != self.n_groups:
                raise ValueError("stratify must have one value per group.")
            stratify = pd.Series(np.asarray(stratify), index=self.obs_names)
        if groups is not None:
            if groups not in self.obs.columns:
                raise KeyError(f"'{groups}' not found in the group table.")
            groups = self.obs[groups]
        return (
            None if stratify is None else _utils.missing_as_na(stratify),
            None if groups is None else _utils.missing_as_na(groups),
        )

    def _split_units(self, stratify, groups) -> tuple[np.ndarray, np.ndarray]:
        stratify, groups = self._split_values(stratify, groups)
        if stratify is None:
            strata = np.zeros(self.n_groups, dtype=np.int64)
        else:
            strata = pd.factorize(stratify, use_na_sentinel=False)[0]
        if groups is None:
            return np.arange(self.n_groups), strata
        block_codes = pd.factorize(groups)[0]
        alone = block_codes < 0
        block_codes[alone] = block_codes.max() + 1 + np.arange(alone.sum())
        block_strata = np.array([np.bincount(strata[block_codes == b]).argmax() for b in range(block_codes.max() + 1)])
        return block_codes, block_strata

    def train_test_split(
        self,
        test_size: float | int = 0.2,
        *,
        stratify: str | Sequence | None = None,
        groups: str | None = None,
        random_state: int | None = 0,
    ) -> tuple[HiData, HiData]:
        """Split groups into a training and a test set.

        All subgroups and cells of a group land on the same side, so models are evaluated on unseen
        groups. With ``groups``, groups that share a site, study or family stay together.

        Parameters
        ----------
        test_size
            Fraction of groups, or of the blocks of groups given by ``groups``, held out for testing, or
            their number when an integer.
        stratify
            Group table column, or one value per group, whose proportions are kept in both sets. A
            ``pandas.Series`` is matched to the groups by name, an array or list by position. With
            ``groups`` each block counts with its most frequent value. Groups whose value is missing, or
            a string such as ``"nan"`` or ``"NA"``, form a stratum of their own.
        groups
            Group table column whose values, such as sites, studies or families, define blocks of groups
            that are never split. A group whose value is missing is a block of its own.
        random_state
            Seed of the random generator.

        Returns
        -------
        tuple of HiData
            Training and test set.
        """
        unit_of_group, strata = self._split_units(stratify, groups)
        n_units = strata.size
        n_test = int(test_size) if isinstance(test_size, int | np.integer) else int(round(test_size * n_units))
        if not 0 < n_test < n_units:
            raise ValueError(f"test_size gives {n_test} test units out of {n_units}.")
        sizes = np.bincount(strata)
        quota = sizes * n_test / n_units
        n_per_stratum = np.floor(quota).astype(int)
        remainder = n_test - n_per_stratum.sum()
        n_per_stratum[np.argsort(-(quota - n_per_stratum), kind="stable")[:remainder]] += 1

        rng = np.random.default_rng(random_state)
        test_units = np.zeros(n_units, dtype=bool)
        for stratum, n_take in enumerate(n_per_stratum):
            members = np.flatnonzero(strata == stratum)
            test_units[rng.choice(members, size=n_take, replace=False)] = True
        test = test_units[unit_of_group]
        return self._subset(G_obs=~test), self._subset(G_obs=test)

    def kfold(
        self,
        n_splits: int = 5,
        *,
        stratify: str | Sequence | None = None,
        groups: str | None = None,
        random_state: int | None = 0,
    ) -> Iterator[tuple[HiData, HiData]]:
        """Iterate over cross validation folds of groups.

        Parameters
        ----------
        n_splits
            Number of folds. Every group is in the test set of exactly one fold.
        stratify
            Group table column, or one value per group, balanced across folds. A ``pandas.Series`` is
            matched to the groups by name, an array or list by position. Groups whose value is
            missing, or a string such as ``"nan"`` or ``"NA"``, form a stratum of their own.
        groups
            Group table column whose values, such as sites or studies, define blocks of groups that are
            never split across folds. A group whose value is missing is a block of its own.
        random_state
            Seed of the random generator.

        Yields
        ------
        tuple of HiData
            Training and test set of every fold.
        """
        unit_of_group, strata = self._split_units(stratify, groups)
        if not 2 <= n_splits <= strata.size:
            raise ValueError(f"n_splits must be between 2 and the number of units ({strata.size}).")
        rng = np.random.default_rng(random_state)
        folds = np.empty(strata.size, dtype=np.int64)
        offset = 0
        for stratum in range(strata.max() + 1):
            members = rng.permutation(np.flatnonzero(strata == stratum))
            folds[members] = (np.arange(members.size) + offset) % n_splits
            offset += members.size
        group_folds = folds[unit_of_group]
        for fold in range(n_splits):
            yield self._subset(G_obs=group_folds != fold), self._subset(G_obs=group_folds == fold)

    def write_h5_hd(self, path: str) -> None:
        """Write to an HDF5 file.

        Parameters
        ----------
        path
            Destination, conventionally ending in ``.hd.h5``.
        """
        self._check_not_reading_from(path)
        with h5py.File(str(path), "w") as f:
            write_group(self, f)

    def write_zarr_hd(self, path: str, x_chunks: tuple[int, int] | dict[str, tuple[int, int]] | None = None) -> None:
        """Write to a Zarr store.

        Parameters
        ----------
        path
            Destination, conventionally ending in ``.hd.zarr``.
        x_chunks
            Chunk shape for a dense ``X`` and dense layers, for example ``(4096, 4096)``, or a dict
            ``{"G": ..., "C": ...}`` to set the levels independently. Without it a dask array keeps
            its own chunks and a plain array is chunked with at most 4096 entries per axis. An EHRData
            level is written by ehrdata with its own chunks, and ``x_chunks`` for it is ignored with a
            warning.
        """
        self._check_not_reading_from(path)
        f = zarr.open(str(path), mode="w")
        write_group(self, f, zarr_path=str(path), x_chunks=x_chunks)

    def _check_not_reading_from(self, path) -> None:
        if any(reads_from(level, path) for level in (self._G, self._S, self._C) if level is not None):
            raise ValueError(
                f"This HiData reads its matrices from {path}, which writing would erase first. Write it to "
                "another path, or read it into memory with lazy=False."
            )

    def write_hd(self, path: str, fmt: Literal["h5", "zarr"] | None = None) -> None:
        """Write to disk, choosing the format from the extension unless ``fmt`` is given.

        Parameters
        ----------
        path
            Destination ending in ``.h5`` or ``.zarr``. Without an extension, ``.hd.h5`` or ``.hd.zarr``
            is appended according to ``fmt``.
        fmt
            ``"h5"`` or ``"zarr"``.
        """
        path = str(path)
        if fmt is None:
            if path.endswith(".h5"):
                fmt = "h5"
            elif path.endswith(".zarr"):
                fmt = "zarr"
            else:
                raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")
        if fmt == "h5":
            self.write_h5_hd(path if path.endswith(".h5") else path + ".hd.h5")
        elif fmt == "zarr":
            self.write_zarr_hd(path if path.endswith(".zarr") else path + ".hd.zarr")
        else:
            raise ValueError("Unknown format: use 'h5' or 'zarr'.")

    def __repr__(self) -> str:
        return render(self)

    def _repr_html_(self) -> str:
        return render(self, html=True)

    def __str__(self) -> str:
        n_groups, n_group_vars, n_cells, n_cell_vars = self.shape
        subgroups = "" if self._S is None else f"n_S_obs={self.n_subgroups:,}, "
        return (
            f"{self.__class__.__name__}"
            f"(n_G_obs={n_groups:,}, n_G_vars={n_group_vars:,}, {subgroups}"
            f"n_C_obs={n_cells:,}, n_C_vars={n_cell_vars:,})"
        )


def _indexed_by(obs: pd.DataFrame, key: str, where: str) -> pd.DataFrame:
    """A copy of ``obs`` indexed by the identifiers as text, with a ``key`` column of the same text if it has one."""
    if key in obs.columns and obs.index.name != key:
        obs = obs.set_index(key, drop=False)
    elif obs.index.name is None:
        obs = obs.rename_axis(key)
    elif obs.index.name != key:
        raise ValueError(f"'{key}' must be {where}")
    obs = obs.copy()
    obs.index = pd.Index(np.asarray(_utils.as_text(obs.index.to_series()), dtype=object), name=key)
    if key in obs.columns:
        obs[key] = np.asarray(obs.index, dtype=object)
    return obs


def _detached(level: AnnData | MuData, obs: pd.DataFrame | None = None) -> AnnData | MuData:
    """A new object with the data of ``level`` and ``obs``, so that changes to it never reach ``level``."""
    obs = level.obs.copy() if obs is None else obs
    if type(level) is AnnData:
        return _utils.shallow_with_obs(level, obs)
    new = _utils.copy_container(level)
    new.obs = obs
    return new


def _match_codes(labels: pd.Series, names: pd.Index, units: str, key: str, checked: bool) -> np.ndarray:
    """Position of every label in ``names``; unless ``checked``, warn about the missing labels."""
    if checked:
        return _utils.codes_for(labels, names)
    codes, missing = _utils.codes_and_missing(labels, names)
    _log_missing_labels(labels, units, key, missing=missing)
    return codes


def _drop_missing_names(level: AnnData | MuData, units: str, key: str) -> AnnData | MuData:
    """``level`` without the rows whose name is missing or a string such as ``"nan"``, with a warning."""
    names = level.obs_names.to_series()
    _log_missing_labels(names, units, key)
    missing = _utils.is_missing_label(names)
    return level[~missing] if missing.any() else level


def _category_codes(groups: pd.Series) -> tuple[np.ndarray, pd.Index]:
    """Code of the cell group of every cell and the cell group names. Cells with a missing one form the cell group ``"nan"``."""
    missing = _utils.is_missing_label(groups)
    groups = groups.where(~missing)
    categories = _observed_categories(groups)
    codes = pd.Categorical(groups, categories=categories).codes.astype(np.int64)
    if missing.any():
        codes[missing] = len(categories)
        categories = categories.astype(str).append(pd.Index([MISSING_GROUP]))
    return codes, categories


def _observed_categories(groups: pd.Series) -> pd.Index:
    if isinstance(groups.dtype, pd.CategoricalDtype):
        present = np.bincount(groups.cat.codes[groups.cat.codes >= 0], minlength=len(groups.cat.categories)) > 0
        return groups.cat.categories[present]
    return pd.Index(sorted(pd.unique(groups.dropna())))


def _copy_matrices(source: AnnData, target: AnnData) -> None:
    """Copy ``obsm`` and ``obsp`` from ``source`` to ``target``, matching rows by name."""
    if not len(source.obsm) and not len(source.obsp):
        return
    aligned = source[target.obs_names]
    for key in aligned.obsm:
        target.obsm[key] = aligned.obsm[key]
    for key in aligned.obsp:
        target.obsp[key] = aligned.obsp[key]


def _carry_over_matrices(previous: AnnData | MuData | None, new: AnnData) -> None:
    if previous is None:
        return
    keys_m = [k for k in previous.obsm if k not in new.obsm]
    keys_p = [k for k in previous.obsp if k not in new.obsp]
    if not keys_m and not keys_p:
        return
    if not new.obs_names.isin(previous.obs_names).all():
        warnings.warn(f"New units in the table, dropping aligned matrices {keys_m + keys_p}.", stacklevel=3)
        return
    aligned = previous[new.obs_names]
    for key in keys_m:
        new.obsm[key] = aligned.obsm[key]
    for key in keys_p:
        new.obsp[key] = aligned.obsp[key]
