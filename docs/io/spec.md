# HiData specification

Building on top of the {doc}`AnnData specification <anndata:fileformat-prose>` and the MuData specification, this
document describes the `HiData` on-disk format. HDF5 files and Zarr stores share the same layout, except for
[EHRData levels](#ehrdata). The examples below use an HDF5 file with three levels.

```{image} ../_static/img/hidata_schema.png
:alt: HiData on-disk layout
:width: 680px
```

```python-console
>>> import h5py
>>> f = h5py.File("cohort.hd.h5")
>>> list(f.keys())
['C', 'G', 'S']
```

```{contents}
:local:
:depth: 3
```

## `.attrs`

The root group identifies the file and names the identifier columns.

```python-console
>>> dict(f.attrs)
{'encoding-type': 'hidata', 'encoding-version': '0.1.0', 'group_id': 'patient_id', 'subgroup_id': 'sample_id', 'var_dims_to_sync': []}
```

`subgroup_id` is only present when the object has a subgroup level. `var_dims_to_sync` lists keys of the group level
`obsm` whose columns follow the variables of `C`. Unstructured annotations of the object are stored as `uns/<key>`
next to the levels.

## `G`, `S` and `C`

Every level is stored in its own HDF5 or Zarr group with the AnnData or MuData encoding, so anndata reads it without
HiData. An EHRData level of a Zarr store keeps its AnnData one group deeper, see [EHRData](#ehrdata).

```python-console
>>> dict(f["C"].attrs)
{'encoding-type': 'anndata', 'encoding-version': '0.1.0'}
>>> import anndata
>>> anndata.io.read_elem(f["S"])
AnnData object with n_obs × n_vars = 4 × 0
    obs: 'patient_id', 'day'
    obsm: 'pseudobulk'
```

`G` has one row per group and `C` one row per cell. `S`, when present, has one row per subgroup and a column named
after `group_id` with the group of every subgroup. Without `S`, `C.obs` has a column named after `group_id`, and with
`S` a column named after `subgroup_id`.

### Tables

A group or subgroup level given as a table is stored as an AnnData without variables, with the table as `obs` and the
matrices aligned to the units in `obsm` and `obsp`. The HDF5 or Zarr group carries the attribute `hidata-table`, and
the reader returns the table itself.

```python-console
>>> dict(f["S"].attrs)
{'encoding-type': 'anndata', 'encoding-version': '0.1.0', 'hidata-table': True}
```

### MuData

A MuData level is stored in the `.h5mu` layout with `encoding-type` `"MuData"`, its modalities below `mod/`, and
`obsmap` and `varmap`. It is written as it is, so the global `obs` and `var` keep their columns.

### EHRData

An EHRData group or subgroup level is written with {func}`ehrdata.io.write_elem` and read with
{func}`ehrdata.io.read_elem`, so it has ehrdata's own layout and ehrdata reads it without HiData. The layout differs
between the formats:

- In an HDF5 file, the level has the `.h5ed` layout: the AnnData slots at the top level of its HDF5 group, the time
  axis as a dataframe in `tem`, and the attributes `ehrdata-encoding-type` `"ehrdata"` and `ehrdata-encoding-version`
  next to anndata's own. anndata reads the level as an AnnData.
- In a Zarr store, the level has the layout of {func}`ehrdata.io.write_zarr`: the AnnData in the subgroup
  `anndata` of its Zarr group, the time axis in `tem`, and the attributes `encoding-type` `"ehrdata"` and
  `ehrdata-encoding-version`. anndata reads the Zarr group `anndata`.

```python-console
>>> dict(f["G"].attrs)
{'ehrdata-encoding-type': 'ehrdata', 'ehrdata-encoding-version': '0.3.0', 'encoding-type': 'anndata', 'encoding-version': '0.1.0'}
>>> import ehrdata
>>> edata = ehrdata.io.read_elem(f["G"])
```

Three dimensional `X` and layers, sparse ones included, `object` arrays holding text, and columns that anndata cannot
write, such as datetimes, are stored the way ehrdata stores them. anndata cannot read a level that holds a sparse
three dimensional layer. HiData reads an EHRData level into memory, also from a Zarr store, with the defaults of
{func}`ehrdata.io.read_elem`.

## `obsm_columns`

AnnData writes a dataframe column by column. A pseudobulk profile with one column per gene would therefore become
tens of thousands of arrays, and in a Zarr store as many files. A numeric dataframe with a single dtype in `obsm` of
the group or subgroup level is stored as one array, and its column names in `obsm_columns` next to `obsm`. For
an EHRData level of a Zarr store that is `anndata/obsm_columns`.

```python-console
>>> f["S"]["obsm"]["pseudobulk"]
<HDF5 dataset "pseudobulk": shape (4, 4), type "<f8">
>>> f["S"]["obsm_columns"]["pseudobulk"][:]
array([b'CD14', b'CD3E', b'MS4A1', b'NKG7'], dtype=object)
```

HiData turns the array back into a dataframe indexed by the units. anndata and ehrdata ignore `obsm_columns`
and read the array.

## Order and links

Subgroups are stored grouped by group in the row order of `G`, and cells grouped by subgroup, or by group, in the row
order of the level above. The cells of every unit are therefore one contiguous block of `C`. {func}`hidata.validate`
checks these rules on an object in memory and raises {class}`hidata.HiDataSchemaError` on the first violation.

## Versions

`encoding-version` follows `MAJOR.MINOR.PATCH`. The `encoding-type` is checked strictly, and an unknown
`encoding-version` is read with a warning.

| Version | Changes |
|---|---|
| `0.1.0` | Levels `G`, `S` and `C`; table levels with `hidata-table`; EHRData levels in ehrdata's layout; `obsm_columns`; MuData levels in Zarr stores |

## Reading donordata stores

HiData continues donordata under a new name. Files and stores with `encoding-type` `"donordata"` are read as well,
and writing the object again stores it in the HiData format. Their layout is the one described above with the
donordata names:

| HiData | donordata |
|---|---|
| `encoding-type` `"hidata"` | `encoding-type` `"donordata"` |
| level `G` | level `D` |
| attribute `group_id` | attribute `donor_id` |
| attribute `subgroup_id` | attribute `sample_id` |
| attribute `hidata-table` | attribute `donordata-table` |

| donordata version | Changes |
|---|---|
| `0.3.0` | Written by donordata 0.1.0. Levels `D`, `S` and `C`; table levels with `donordata-table`; EHRData levels in ehrdata's layout; `obsm_columns`; MuData levels in Zarr stores |
| `0.2.0` | Written by cellink. Donor level `G`, which could also be a dataframe with `obsm` and `obsp` at the root |
| `0.1.0` | Written by cellink. Donor level `G` and cells `C`, both AnnData or MuData |

Files of the versions 0.1.0 and 0.2.0, written by cellink, are read with a warning.
