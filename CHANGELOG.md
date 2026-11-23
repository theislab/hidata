# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0]

First release of HiData, which continues donordata 0.1.0 under a new name. The levels are groups, optional
subgroups and cells, where a group can be a donor, a patient, a target gene or a compound, and a subgroup a sample, a
time point, a guide or a dose. The behaviour is that of donordata 0.1.0; the names and the on-disk format change.

### Changed

- The package `donordata` is `hidata` and `DonorData` is `HiData`. `DonorDataSchemaError`, `DONORDATA_ENCODING_TYPE`,
  `DONORDATA_ENCODING_VERSION`, `register_donordata_namespace` and `DAnn` are `HiDataSchemaError`,
  `HIDATA_ENCODING_TYPE`, `HIDATA_ENCODING_VERSION`, `register_hidata_namespace` and `GAnn`.
- The donor level `D` is the group level `G`, and the sample level `S` is the subgroup level `S`: `G=`, `group_id` and
  `subgroup_id` in the constructor and `from_anndata`, `G_obs` and `G_var` in `sel`, `G_type`, and the level names
  `"group"` and `"subgroup"` in `level=`, `levels`, `level_names` and `infer_levels`. `get_df` takes the prefix `G:`.
- `n_donors`, `n_samples`, `donor_codes`, `sample_codes`, `sample_donor_codes`, `n_cells_per_donor`,
  `n_cells_per_sample` and `n_samples_per_donor` are `n_groups`, `n_subgroups`, `group_codes`, `subgroup_codes`,
  `subgroup_group_codes`, `n_cells_per_group`, `n_cells_per_subgroup` and `n_subgroups_per_group`.
- `filter_donors`, `filter_samples` and `filter_donors_by_cells` are `filter_groups`, `filter_subgroups` and
  `filter_groups_by_cells`, and `nest(keep_in_samples=)` is `nest(keep_in_subgroups=)`.
- `pp.donor_level_obs_filter` and `pp.donor_level_var_filter` are `pp.group_level_obs_filter` and
  `pp.group_level_var_filter`, with `group_level_key` and `group_level_values`. The object passed to `concat`,
  `validate`, the `pp` functions and `MILDataset` is called `hds` or `hd`.
- `MILDataset` takes `group_layer`, `group_labels_key`, `group_batch_key`, `group_cat_covs_key`,
  `group_cont_covs_key`, `group_indices_key` and `split_groups`, has `group_names` and `hd`, and yields `group_id`,
  `group_x`, `group_y`, `group_batch`, `group_cat_covs`, `group_cont_covs` and `group_indices`.
- The fields of `GAnn` and their default names refer to groups, for example `GAnn.group` is `"group_id"` and
  `GAnn.GROUP_LABELS_KEY` is `"group_labels"`, and `CAnn.CELL_DONOR_KEY` is `CAnn.CELL_GROUP_KEY`. The default
  `group_id` is therefore `"group_id"`.
- `read_dd`, `read_h5_dd`, `read_zarr_dd`, `write_dd`, `write_h5_dd` and `write_zarr_dd` are `read_hd`, `read_h5_hd`,
  `read_zarr_hd`, `write_hd`, `write_h5_hd` and `write_zarr_hd`, and files end in `.hd.h5` and `.hd.zarr`.
- The on-disk format 0.1.0 has the `encoding-type` `"hidata"`, the levels `G`, `S` and `C`, the attributes `group_id`
  and `subgroup_id`, and `hidata-table` on table levels. `x_chunks` takes the key `"G"`.
- The representation reads `HiData(n_groups=..., n_subgroups=..., n_cells=...)`, and `str` gives `n_G_obs` and
  `n_G_vars`.

### Added

- Reading of files and stores written by donordata 0.1.0, and of the older files of cellink that donordata reads.
  Writing such an object again stores it in the HiData format.

## donordata releases

HiData was released as donordata before. These entries use the donordata names.

### donordata [0.1.0] - 2026-11-20

First release of DonorData as a package of its own, split from cellink.

#### Changed

- Donors, samples and cells whose donor or sample identifier is missing are dropped with a warning that names
  them. Before, a missing identifier became a donor or sample called `"nan"` or `"None"`. A string that stands for
  a missing value (`""`, `"NA"`, `"N/A"`, `"<NA>"`, `"nan"`, `"NaT"`, `"None"` or `"null"`, in any case) counts as
  missing too and has to be renamed if it is a real name.
- `nest` and `collapse` drop donors whose grouping value is missing, with a warning.
- `train_test_split`, `kfold` and `subsample_cells` treat a missing `stratify`, `groups` or `groupby` value as a group
  of its own.
- `composition` and `pseudobulk` count cells whose `groupby` value is missing as the group `"nan"`;
  `filter_donors_by_cells` only requires that group when it is listed in `groups`.
- `train_test_split` and `kfold` match a `stratify` given as a `pandas.Series` to the donors by name. Before, they
  read it by position, which assigned wrong classes when the Series was in another order than the donors.
- Selections, splits, `nest`, `flatten`, `collapse` and `subsample_cells` return objects that do not share levels or
  `uns` with the original.
- The constructor and the level setters leave their inputs unchanged; a rejected assignment leaves the object
  unchanged; replacing a table keeps the matrices of the remaining units.
- Identifiers are text in the index and the identifier column; whole-number floats match integer identifiers.
- `from_anndata` gives a cell without a donor the donor of its sample.
- MuData modalities are matched by name in `aggregate`, `pseudobulk`, `get_df` and `MILDataset`.
- `get_df` gives one column per time point for an EHRData variable.
- `concat(join="outer")` keeps all columns of donor and sample matrices; `concat` rejects EHRData levels.
- The `filter_*` methods accept a set of names.
- Requires Python 3.12 or newer and anndata 0.13.1 or newer, with the numpy (2.1), pandas (2.3), scipy (1.14) and h5py
  (3.11) versions that anndata 0.13 needs.
- Selections and the constructor keep the levels as views instead of copying them. Assigning `X` of a level copies
  that level first, as anndata 0.13 does for every view, so the original never changes.
- Views of a donor-level AnnData keep all categories of `obs` and `var`, so selecting donors, cells or genes costs no
  more than reading the selected genotypes.

#### Added

- `DonorData` with a donor level `D`, an optional sample level `S` and the cells `C`, where each level can be a
  table, an `AnnData`, a `MuData`, or for donors and samples an `EHRData`.
- Selection at every level with `filter_donors`, `filter_samples`, `filter_cells`, `filter_donors_by_cells` and
  `subsample_cells`, and `coverage` and `complete_cases` for `MuData` cells with assays on different cells.
- `get_df`, `infer_levels`, `push_obs`, `pull_obs`, `push_obsm` and `to_anndata` to move information between
  levels.
- `aggregate`, `pseudobulk`, `composition` and `cell_weights`.
- `nest`, `flatten` and `collapse` to change the levels.
- `train_test_split` and `kfold` over donors, with stratification and groups.
- `donordata.io` with `read_dd`, `read_h5_dd` and `read_zarr_dd`, and the on-disk format 0.3.0. `read_zarr_dd` and
  `read_h5_dd(lazy=True)` keep `X` and layers on disk; `read_zarr_dd(lazy=False)` reads them into memory.
  Writing to a file or store that a lazily read object still reads from raises an error.
- `concat`, `validate` and `register_donordata_namespace`. `concat` returns sparse `obsm` matrices as scipy sparse
  arrays (`csr_array`), since scipy 2 deprecates sparse matrices (`csr_matrix`).
- `donordata.ml.MILDataset` with bags per donor or per sample.
- Reading of files in the versions 0.1.0 and 0.2.0 written by cellink. Their donor level `G` is read into `D`
  with a warning.

#### Fixed

- `sel`, `dd[...]`, `D_var` and `C_var` match a boolean `pandas.Series` to the axis by name, as `filter_donors`,
  `filter_samples` and `filter_cells` already do. Before, they applied it by position, which selected the wrong
  samples or cells when the Series was built from the input tables, because `DonorData` reorders samples and
  cells. A Series whose index does not contain the names of the axis, such as the default index 0, 1, 2, ..., now
  raises a `ValueError` instead of selecting by position, also when the cells have no real names and are
  called `"0"`, `"1"`, .... Boolean arrays and lists are still applied by position. A missing entry (NA) of a
  mask counts as `False`.
- `dd[mask, ...]` with an ellipsis and a boolean array or Series no longer raises an error.
- `filter_donors`, `filter_samples` and `filter_cells` raise a `KeyError` for a name that does not exist, as `sel`
  and `dd[...]` do. Before, they ignored it. They raise a `TypeError` for integers, which they used to read as
  names, and for slices. Names are given as text; positions are selected with `dd[...]` or `sel`.
