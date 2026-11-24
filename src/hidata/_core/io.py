from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import anndata as ad
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata._types import StorageType
from anndata.compat import _read_attr
from anndata.experimental import read_dispatched
from anndata.io import read_elem, sparse_dataset
from mudata import MuData

from hidata._core._utils import build_mudata
from hidata._core._write import ehrdata_io, is_ehrdata_group, slots_group
from hidata._core.fields import GAnn
from hidata._core.hidata import HiData, _copy_matrices
from hidata._core.schema import (
    DONORDATA_ENCODING_TYPE,
    DONORDATA_ENCODING_VERSIONS,
    HIDATA_ENCODING_TYPE,
    HIDATA_ENCODING_VERSION,
    READABLE_ENCODING_VERSIONS,
)


def _is_matrix_elem(elem_name: str) -> bool:
    return elem_name.endswith("/X") or elem_name.rsplit("/", 1)[0].endswith("/layers")


_ANNDATA_SLOTS = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "raw")


def _read_columns(group, columns) -> pd.DataFrame:
    """Read the index and ``columns`` of a DataFrame stored by anndata."""
    missing = [c for c in columns if c not in group]
    if missing:
        raise KeyError(f"Columns {missing} not found in {group.name}.")
    index_key = group.attrs["_index"]
    frame = pd.DataFrame({c: read_elem(group[c]) for c in columns}, index=pd.Index(read_elem(group[index_key])))
    if index_key != "_index":
        frame.index.name = index_key
    return frame


def _lazy_callback(func, elem_name: str, elem, iospec, var_columns=None):
    """``read_dispatched`` callback that keeps ``X`` and layers on disk, dense ones as dask arrays."""
    if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
        callback = partial(_lazy_callback, var_columns=var_columns)
        return ad.AnnData(**{k: read_dispatched(v, callback) for k, v in dict(elem).items() if k in _ANNDATA_SLOTS})
    if var_columns is not None and elem_name.endswith("/var") and iospec.encoding_type == "dataframe":
        return _read_columns(elem, var_columns)
    if _is_matrix_elem(elem_name) and iospec.encoding_type == "array":
        if isinstance(elem, zarr.Array):
            return da.from_zarr(elem)
        return da.from_array(elem, chunks=elem.chunks or tuple(min(n, 4096) for n in elem.shape), lock=True)
    if _is_matrix_elem(elem_name) and iospec.encoding_type in ("csr_matrix", "csc_matrix"):
        return sparse_dataset(elem)
    if _is_matrix_elem(elem_name):
        return read_elem(elem)
    return func(elem)


def _read_anndata(group, lazy: bool, var_columns=None):
    if lazy:
        return read_dispatched(group, callback=partial(_lazy_callback, var_columns=var_columns))
    adata = read_elem(group)
    if var_columns is not None:
        adata.var = _read_columns(group["var"], var_columns)
    return adata


def _read_mudata(group: StorageType, lazy: bool = False) -> MuData:
    """Read a MuData stored in a group of a larger HDF5 file or Zarr store."""
    mods_group = group["mod"]
    order = [str(m) for m in _read_attr(mods_group.attrs, "mod-order")] if "mod-order" in mods_group.attrs else []
    names = [m for m in order if m in mods_group] + [m for m in mods_group.keys() if m not in order]
    mods = {name: _read_anndata(mods_group[name], lazy) for name in names}
    slots = ("obs", "var", "obsm", "varm", "obsp", "varp", "uns", "obsmap", "varmap")
    kwargs = {slot: read_elem(group[slot]) for slot in slots if slot in group}
    axis = int(group.attrs.get("axis", 0))
    return build_mudata(lambda: MuData(mods, axis=axis, **kwargs))


def _read_side(group, lazy: bool, var_columns=None):
    encoding = group.attrs.get("encoding-type")
    if encoding == "MuData":
        return _read_mudata(group, lazy)
    if encoding == "anndata":
        return _read_anndata(group, lazy, var_columns)
    if encoding == "dataframe":
        return read_elem(group)
    raise ValueError(f"Unknown encoding type {encoding!r} for {group.name}")


def _unpack_obsm(group, container) -> None:
    """Turn the arrays that were written for numeric DataFrames in ``obsm`` back into DataFrames."""
    if "obsm_columns" not in group:
        return
    for entry in group["obsm_columns"].keys():
        names = np.asarray(read_elem(group["obsm_columns"][entry])).astype(object)
        columns = pd.Index([n.decode() if isinstance(n, bytes) else str(n) for n in names])
        values = np.asarray(container.obsm[entry])
        container.obsm[entry] = pd.DataFrame(values, index=container.obs_names, columns=columns)


def _read_level(f, key: str, lazy: bool, var_columns=None, table_attr: str = "hidata-table"):
    """Return the object to hand to HiData for a level, and the AnnData holding a table level."""
    group = f[key]
    if group.attrs.get(table_attr, False):
        shell = read_elem(group)
        _unpack_obsm(group, shell)
        return shell.obs, shell
    if is_ehrdata_group(group):
        container = ehrdata_io(f"Reading the EHRData level {key}").read_elem(group)
    elif "tem" in group:
        raise ValueError(f"{group.name} holds an EHRData in the layout of a development version; write it again.")
    else:
        container = _read_side(group, lazy, var_columns)
    if isinstance(container, AnnData):
        _unpack_obsm(slots_group(group), container)
    return container, None


def _layout(f) -> tuple[str, str, str, str, str]:
    """Level key, table attribute, identifier attributes and default ``group_id`` of a HiData or donordata store."""
    encoding_type = f.attrs.get("encoding-type")
    encoding_version = f.attrs.get("encoding-version")
    if encoding_type == HIDATA_ENCODING_TYPE:
        if encoding_version is not None and encoding_version not in READABLE_ENCODING_VERSIONS:
            warnings.warn(
                f"HiData file has encoding-version={encoding_version!r}, this version of HiData writes "
                f"{HIDATA_ENCODING_VERSION!r}. Reading anyway, but the result may be incomplete.",
                stacklevel=4,
            )
        return "G", "hidata-table", "group_id", "subgroup_id", GAnn.group
    if encoding_type is not None and encoding_type != DONORDATA_ENCODING_TYPE:
        raise ValueError(
            f"Expected encoding-type={HIDATA_ENCODING_TYPE!r} at the top level of a HiData file, "
            f"found {encoding_type!r}. This file was not written by HiData."
        )
    if encoding_version is not None and encoding_version not in DONORDATA_ENCODING_VERSIONS:
        warnings.warn(
            f"donordata file has encoding-version={encoding_version!r}, HiData reads donordata files of the "
            f"versions {', '.join(DONORDATA_ENCODING_VERSIONS)}. Reading anyway, but the result may be incomplete.",
            stacklevel=4,
        )
    if "D" in f:
        return "D", "donordata-table", "donor_id", "sample_id", "donor"
    warnings.warn(
        f"This file has the layout of donordata files of version {encoding_version or '0.1.0'}, written by "
        "cellink. It is read into a HiData, and writing the object again stores it in the current format.",
        UserWarning,
        stacklevel=4,
    )
    return "G", "donordata-table", "donor_id", "sample_id", "donor"


def _read_hd(f, lazy: bool = False, var_columns=None) -> HiData:
    group_key, table_attr, group_attr, subgroup_attr, default_group_id = _layout(f)
    G, G_shell = _read_level(f, group_key, lazy, var_columns, table_attr)
    S, S_shell = _read_level(f, "S", lazy, table_attr=table_attr) if "S" in f else (None, None)
    C = _read_side(f["C"], lazy)
    if G_shell is not None:
        obsm, obsp = dict(G_shell.obsm), dict(G_shell.obsp)
    else:
        obsm = read_elem(f["obsm"]) if "obsm" in f else None
        obsp = read_elem(f["obsp"]) if "obsp" in f else None
    uns = {key: read_elem(f["uns"][key]) for key in f["uns"].keys()} if "uns" in f else {}
    subgroup_id = f.attrs.get(subgroup_attr)

    hd = HiData(
        G=G,
        S=S,
        C=C,
        group_id=f.attrs.get(group_attr, default_group_id),
        subgroup_id=None if subgroup_id is None else str(subgroup_id),
        var_dims_to_sync=list(f.attrs.get("var_dims_to_sync", [])),
        uns=uns,
        obsm=obsm,
        obsp=obsp,
    )
    if S_shell is not None:
        _copy_matrices(S_shell, hd._S)
    return hd


def read_h5_hd(path: str | Path, lazy: bool = False, var_columns: list[str] | None = None) -> HiData:
    """Read a HiData from an HDF5 file.

    Parameters
    ----------
    path
        File written by :meth:`HiData.write_h5_hd`, or by donordata.
    lazy
        Keep ``X`` and layers on disk, dense ones as dask arrays and sparse ones as backed sparse
        matrices, as :func:`read_zarr_hd` does. The file stays open while they are in use.
    var_columns
        Columns of ``G.var`` to read, for example ``["chrom", "pos"]`` for genotypes. All by default.

    Returns
    -------
    HiData
    """
    if lazy:
        return _read_hd(h5py.File(str(path), "r"), lazy=True, var_columns=var_columns)
    with h5py.File(str(path), "r") as f:
        return _read_hd(f, var_columns=var_columns)


def read_zarr_hd(path: str | Path, var_columns: list[str] | None = None, lazy: bool = True) -> HiData:
    """Read a HiData from a Zarr store.

    ``X`` and layers stay on disk, dense ones as dask arrays and sparse ones as backed sparse matrices,
    so selecting a subset, for example a single cell type, only reads that subset, and
    :meth:`HiData.aggregate` reads them block by block. A group or subgroup level stored as an EHRData
    is read into memory with :func:`ehrdata.io.read_elem`.

    Parameters
    ----------
    path
        Store written by :meth:`HiData.write_zarr_hd`, or by donordata.
    var_columns
        Columns of ``G.var`` to read, for example ``["chrom", "pos"]`` for genotypes. All by default.
    lazy
        Read ``X`` and layers into memory with ``False``, for tools that need in-memory matrices.

    Returns
    -------
    HiData
    """
    return _read_hd(zarr.open(str(path), mode="r"), lazy=lazy, var_columns=var_columns)


def read_hd(path: str | Path, fmt: str | None = None, lazy: bool | None = None) -> HiData:
    """Read a HiData from HDF5 or Zarr, choosing the format from the extension.

    Files and stores written by donordata are read as well, see the :doc:`specification </io/spec>`.

    Parameters
    ----------
    path
        File or store ending in ``.h5`` or ``.zarr``.
    fmt
        ``"h5"`` or ``"zarr"`` to override the detection.
    lazy
        Keep ``X`` and layers on disk, see :func:`read_h5_hd` and :func:`read_zarr_hd`. Defaults to
        ``False`` for HDF5 and ``True`` for Zarr.

    Returns
    -------
    HiData

    Examples
    --------
    >>> hd = read_hd("cohort.hd.h5")
    >>> hd = read_hd("cohort.hd.zarr")
    """
    path = str(path).rstrip("/")
    if fmt is None:
        if path.endswith(".h5"):
            fmt = "h5"
        elif path.endswith(".zarr"):
            fmt = "zarr"
        else:
            raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")
    if fmt == "h5":
        return read_h5_hd(path, lazy=bool(lazy))
    if fmt == "zarr":
        return read_zarr_hd(path, lazy=True if lazy is None else lazy)
    raise ValueError("Unknown format: use 'h5' or 'zarr'.")
