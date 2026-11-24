# Designing `HiData`

This document outlines design considerations and some technical details about the `HiData` implementation.

## Levels as containers

`HiData` stores data at several levels: groups, optionally subgroups, and cells. Every level is a fully functional
`AnnData` or `MuData` object that can be used on its own, and the levels are linked by identifiers instead of being
merged into one matrix. This follows the container approach of `MuData`, which keeps every modality as an `AnnData`.
A group table plays a similar role to the global `obs` of a `MuData`, one level up. A group can be any unit that groups
cells, for example a donor or patient, a target gene in a CRISPR screen or a compound in a drug screen.

An `EHRData` from ehrdata, an `AnnData` with a time axis, fits the same scheme: it can be the group or the subgroup
level, for example to hold the clinical record of every patient next to the cells.

## The subgroup level

Some cohorts have several samples per patient, for example blood drawn on two days, two tissues or several
stimulation conditions. Without a subgroup level there are two options, and both lose information. With the patient
as the group, the samples are pooled and can no longer be compared. With the sample as the group, patient covariates
are repeated for every sample, and a split can put two samples of the same patient into training and test data.

The optional subgroup level `S` sits between groups and cells. `G` has one row per group, `S` one row per subgroup
with the group of every subgroup, and `C` the cells with the subgroup of every cell. Analyses can use either level,
and splits are always drawn over groups. Without `S`, cells are linked to groups directly and nothing else changes.

## Order of units

Subgroups are stored grouped by group in the row order of `G`, and cells grouped by subgroup, or by group, in the row
order of the level above. The cells of a unit are therefore one contiguous block of `C`, which makes selecting the
cells of a unit, building bags of cells for multiple instance learning and aggregating cells to units simple slicing
or a single pass over integer codes. The order is restored on construction and whenever a level is replaced, and
`validate` checks it.

## Storage

A `HiData` is written with one HDF5 or Zarr group per level, `G`, `S` and `C`, each with the `AnnData` or `MuData` encoding, so
every level can be read with AnnData alone. See the [specification](docs/io/spec.md) for the full layout.
