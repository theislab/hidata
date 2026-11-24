import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import hidata
from hidata import HiData

MISSING_GROUP_IDS = {
    "object None and nan": np.array(["a", "a", None, "b", np.nan], dtype=object),
    "string dtype pd.NA": pd.array(["a", "a", pd.NA, "b", pd.NA], dtype="string"),
    "categorical NaN": pd.Categorical(["a", "a", None, "b", np.nan]),
}


@pytest.mark.parametrize("group_ids", MISSING_GROUP_IDS.values(), ids=MISSING_GROUP_IDS.keys())
def test_cells_with_missing_group_are_dropped(caplog, group_ids):
    obs = pd.DataFrame({"group_id": group_ids}, index=[f"c{i}" for i in range(5)])
    with caplog.at_level("WARNING"):
        hd = HiData.from_anndata(AnnData(np.zeros((5, 1)), obs=obs), "group_id")
    assert list(hd.obs_names) == ["a", "b"]
    assert hd.n_cells == 3
    assert "Dropping 2 cells with a missing 'group_id'" in caplog.text


def test_subgroups_and_cells_with_missing_labels_are_dropped(caplog):
    subgroups = pd.DataFrame({"group_id": ["a", np.nan, "b"]}, index=pd.Index(["s1", "s2", "s3"], name="subgroup_id"))
    obs = pd.DataFrame({"subgroup_id": ["s1", "s2", None, "s3"]}, index=[f"c{i}" for i in range(4)])
    with caplog.at_level("WARNING"):
        hd = HiData(S=subgroups, C=AnnData(np.zeros((4, 1)), obs=obs), subgroup_id="subgroup_id")
    assert list(hd.obs_names) == ["a", "b"]
    assert list(hd.levels["subgroup"].obs_names) == ["s1", "s3"]
    assert list(hd.C.obs_names) == ["c0", "c3"]
    assert "Dropping 1 subgroups with a missing 'group_id'" in caplog.text
    assert "Dropping 1 cells with a missing 'subgroup_id'" in caplog.text


# Strings that stand for a missing value are dropped like missing values, in any case and with spaces around them.
MISSING_STRINGS = [
    "none",
    "None",
    "NONE",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "Null",
    "NA",
    "<NA>",
    "N/A",
    "",
    " nan ",
    "NaT",
]


def test_groups_named_like_missing_values_are_dropped(caplog):
    group_ids = ["a", "b", *MISSING_STRINGS]
    obs = pd.DataFrame({"group_id": group_ids}, index=[f"c{i}" for i in range(len(group_ids))])
    with caplog.at_level("WARNING"):
        hd = HiData.from_anndata(AnnData(np.zeros((len(obs), 1)), obs=obs), "group_id")
    assert list(hd.obs_names) == ["a", "b"]
    assert hd.n_cells == 2
    (record,) = [r for r in caplog.records if "Dropping" in r.message]
    assert record.levelname == "WARNING"
    assert f"Dropping {len(MISSING_STRINGS)} cells with a missing 'group_id'" in record.message
    assert "'c2' ('none')" in record.message  # names the dropped cells and their group IDs
    assert f"and {len(MISSING_STRINGS) - 10} more" in record.message
    assert "rename them" in record.message


@pytest.mark.parametrize("extension", [None, ".hd.h5", ".hd.zarr"])
def test_groups_that_only_resemble_missing_values_are_kept(tmp_path, caplog, extension):
    names = ["Nancy", "nana", "nonexistent", "NA12878", "null_1", "N/A1"]
    obs = pd.DataFrame({"group_id": names}, index=[f"c{i}" for i in range(len(names))])
    with caplog.at_level("WARNING"):
        hd = HiData.from_anndata(AnnData(np.zeros((len(obs), 1)), obs=obs), "group_id")
    if extension is not None:
        hd.write_hd(str(tmp_path / f"x{extension}"))
        hd = hidata.io.read_hd(str(tmp_path / f"x{extension}"))
    assert sorted(hd.obs_names) == sorted(names)
    assert "Dropping" not in caplog.text


def test_group_and_subgroup_rows_named_like_missing_values_are_dropped(caplog):
    groups = pd.DataFrame(index=pd.Index(["a", "None"], name="group_id"))
    subgroups = pd.DataFrame({"group_id": ["a", "a", "None"]}, index=pd.Index(["s1", "NA", "s3"], name="subgroup_id"))
    obs = pd.DataFrame({"subgroup_id": ["s1", "NA", "s3"]}, index=["c0", "c1", "c2"])
    with caplog.at_level("WARNING"):
        hd = HiData(
            G=groups, S=subgroups, C=AnnData(np.zeros((3, 1)), obs=obs), group_id="group_id", subgroup_id="subgroup_id"
        )
    assert list(hd.obs_names) == ["a"]
    assert list(hd.levels["subgroup"].obs_names) == ["s1"]
    assert list(hd.C.obs_names) == ["c0"]
    assert (
        "Dropping 1 groups with a missing 'group_id' (empty, or a value such as NaN, None, 'nan' or 'NA'): 'None'."
        in (caplog.text)
    )
    assert "Dropping 1 subgroups with a missing 'subgroup_id'" in caplog.text
    assert "Dropping 1 subgroups with a missing 'group_id'" in caplog.text and "'s3' ('None')" in caplog.text


def _site_hd(sites):
    groups = pd.DataFrame({"site": sites}, index=pd.Index([f"d{i}" for i in range(len(sites))], name="group_id"))
    obs = pd.DataFrame({"group_id": groups.index}, index=[f"c{i}" for i in range(len(sites))])
    return HiData(G=groups, C=AnnData(np.ones((len(sites), 1)), obs=obs), group_id="group_id")


UNITS = {
    "nest": lambda hd: hd.nest("site"),
    "collapse": lambda hd: hd.collapse("site"),
}


@pytest.mark.parametrize("regroup", UNITS.values(), ids=UNITS.keys())
@pytest.mark.parametrize("missing", [None, np.nan, "NA", "nan", "None", ""], ids=repr)
def test_nest_and_collapse_drop_groups_with_a_missing_unit(caplog, regroup, missing):
    hd = _site_hd(np.array(["X", missing, "X", "Y"], dtype=object))
    with caplog.at_level("WARNING"):
        result = regroup(hd)
    assert list(result.obs_names) == ["X", "Y"]
    assert sorted(result.C.obs_names) == ["c0", "c2", "c3"]
    assert "Dropping 1 groups with a missing 'site'" in caplog.text
    assert "'d1'" in caplog.text  # names the dropped group


@pytest.mark.parametrize("regroup", UNITS.values(), ids=UNITS.keys())
def test_nest_and_collapse_keep_all_groups_without_missing_units(caplog, regroup):
    with caplog.at_level("WARNING"):
        result = regroup(_site_hd(["X", "Nancy", "X", "NA12878"]))
    assert list(result.obs_names) == ["NA12878", "Nancy", "X"]
    assert result.n_cells == 4
    assert "Dropping" not in caplog.text


def _grouped_hd(cell_types, sex=("F", "M", "F", "M")):
    """Four groups with two cells each, a cell type per cell and a sex per group."""
    groups = pd.DataFrame({"sex": list(sex)}, index=pd.Index(["a", "b", "c", "d"], name="group_id"))
    obs = pd.DataFrame(
        {"group_id": np.repeat(groups.index, 2), "cell_type": np.array(cell_types, dtype=object)},
        index=[f"c{i}" for i in range(8)],
    )
    return HiData(G=groups, C=AnnData(np.ones((8, 2)), obs=obs), group_id="group_id")


MISSING_GROUPS = [None, np.nan, "NA", "nan", "None", ""]


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
def test_composition_and_pseudobulk_count_cells_with_a_missing_group_as_nan(caplog, missing):
    hd = _grouped_hd(["T", missing, "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        fractions = hd.composition("cell_type")
        profiles = hd.pseudobulk(groupby="cell_type")
    assert list(fractions.columns) == ["B", "T", "nan"]
    assert fractions.loc["a"].tolist() == [0.0, 0.5, 0.5]
    assert fractions.loc["b"].tolist() == [0.5, 0.5, 0.0]
    assert sorted(profiles.obs_names) == ["a_T", "a_nan", "b_B", "b_T", "c_B", "d_T"]
    assert profiles.obs.loc["a_nan", "n_cells"] == 1
    assert caplog.text == ""


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
def test_filter_groups_by_cells_requires_the_missing_group_only_when_listed(missing):
    hd = _grouped_hd(["T", missing, "T", "B", "B", "B", "T", "T"])
    assert list(hd.filter_groups_by_cells(1, groupby="cell_type").obs_names) == ["b"]
    assert list(hd.filter_groups_by_cells(1, groupby="cell_type", groups=["nan"]).obs_names) == ["a"]


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
@pytest.mark.parametrize("inplace", [False, True], ids=["copy", "inplace"])
def test_subsample_cells_keeps_cells_with_a_missing_group_as_a_group(caplog, missing, inplace):
    hd = _grouped_hd(["T", missing, "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        result = hd.subsample_cells(1, groupby="cell_type", inplace=inplace)
    result = hd if inplace else result
    assert "c1" in result.C.obs_names
    assert result.n_cells == 6
    assert "Dropping" not in caplog.text


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
@pytest.mark.parametrize("as_column", [True, False], ids=["column", "values"])
def test_splits_keep_groups_with_a_missing_stratum(caplog, missing, as_column):
    sex = ["F", "M", missing, "M"]
    hd = _grouped_hd(["T"] * 8, sex=sex)
    stratify = "sex" if as_column else sex
    with caplog.at_level("WARNING"):
        train, test = hd.train_test_split(test_size=1, stratify=stratify)
        folds = list(hd.kfold(3, stratify=stratify))
    assert sorted([*train.obs_names, *test.obs_names]) == ["a", "b", "c", "d"]
    assert sorted(name for _, fold_test in folds for name in fold_test.obs_names) == ["a", "b", "c", "d"]
    assert "Dropping" not in caplog.text


def test_missing_stratify_values_form_one_stratum():
    hd = _grouped_hd(["T"] * 8, sex=["F", None, "nan", "F"])
    assert len(set(hd._split_units("sex", None)[1])) == 2


def test_splits_keep_groups_with_a_missing_block_as_their_own_blocks(caplog):
    hd = _grouped_hd(["T"] * 8)
    hd.obs["site"] = ["x", "NA", None, "x"]
    with caplog.at_level("WARNING"):
        train, test = hd.train_test_split(test_size=1, groups="site", random_state=0)
        units, _ = hd._split_units(None, "site")
    assert sorted([*train.obs_names, *test.obs_names]) == ["a", "b", "c", "d"]
    assert units[0] == units[3] and len({units[0], units[1], units[2]}) == 3
    assert "Dropping" not in caplog.text


NAN_NAMED_GROUPS = {
    "index": lambda: pd.DataFrame({"age": [1.0, 2.0, 3.0]}, index=pd.Index(["a", np.nan, "c"], name="group_id")),
    "column": lambda: pd.DataFrame({"group_id": ["a", np.nan, "c"], "age": [1.0, 2.0, 3.0]}),
    "two missing": lambda: pd.DataFrame({"group_id": ["a", np.nan, None, "c"], "age": [1.0, 2.0, 2.5, 3.0]}),
}


@pytest.mark.parametrize("make_groups", NAN_NAMED_GROUPS.values(), ids=NAN_NAMED_GROUPS.keys())
def test_groups_with_a_missing_name_do_not_collect_cells_without_a_group(caplog, make_groups):
    obs = pd.DataFrame({"group_id": ["a", "a", np.nan, None, "c", "c"]}, index=[f"x{i}" for i in range(6)])
    with caplog.at_level("WARNING"):
        hd = HiData(G=make_groups(), C=AnnData(np.ones((6, 1)), obs=obs), group_id="group_id")
    assert list(hd.obs_names) == ["a", "c"]
    assert list(hd.C.obs_names) == ["x0", "x1", "x4", "x5"]
    assert "Dropping 2 cells with a missing 'group_id'" in caplog.text
    assert "groups with a missing 'group_id'" in caplog.text


def test_subgroups_with_a_missing_name_do_not_collect_cells_without_a_subgroup():
    subgroups = pd.DataFrame({"group_id": ["a", "b", "b"]}, index=pd.Index(["s1", np.nan, "s3"], name="subgroup_id"))
    obs = pd.DataFrame({"subgroup_id": ["s1", np.nan, "s3", None]}, index=[f"c{i}" for i in range(4)])
    hd = HiData(S=subgroups, C=AnnData(np.zeros((4, 1)), obs=obs), subgroup_id="subgroup_id")
    assert list(hd.levels["subgroup"].obs_names) == ["s1", "s3"]
    assert list(hd.C.obs_names) == ["c0", "c2"]


def test_grouping_without_missing_values_does_not_warn(caplog):
    hd = _grouped_hd(["T", "Nancy", "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        assert list(hd.composition("cell_type").columns) == ["B", "Nancy", "T"]
        hd.pseudobulk(groupby="cell_type")
        assert hd.subsample_cells(1, groupby="cell_type").n_cells == 6
        train, test = hd.train_test_split(test_size=2, stratify="sex")
    assert train.n_groups + test.n_groups == 4
    assert "Dropping" not in caplog.text and "Ignoring" not in caplog.text


def test_splits_match_a_stratify_series_by_name():
    hd = _grouped_hd(["T"] * 8, sex=["F", "F", "M", "M"])
    other_order = hd.obs["sex"].loc[["c", "a", "d", "b"]]
    with_extra = pd.concat([other_order, pd.Series(["F"], index=["not a group"])])
    for stratify in ("sex", other_order, with_extra, hd.obs["sex"].to_numpy(), hd.obs["sex"].tolist()):
        for seed in range(8):
            _, test = hd.train_test_split(test_size=2, stratify=stratify, random_state=seed)
            assert sorted(test.obs["sex"]) == ["F", "M"]
            for _, fold_test in hd.kfold(2, stratify=stratify, random_state=seed):
                assert sorted(fold_test.obs["sex"]) == ["F", "M"]


@pytest.mark.parametrize(
    "stratify",
    [
        pd.Series(["F", "F", "M", "M"]),
        pd.Series(["F", "F", "M"], index=["a", "b", "c"]),
        pd.Series(["F", "F", "M", "M", "M"], index=["a", "b", "c", "d", "d"]),
    ],
    ids=["default index", "group missing", "duplicate group"],
)
def test_splits_reject_a_stratify_series_that_cannot_be_matched_by_name(stratify):
    hd = _grouped_hd(["T"] * 8)
    with pytest.raises(ValueError, match="cannot be matched by name"):
        hd.train_test_split(test_size=2, stratify=stratify)
    with pytest.raises(ValueError, match="cannot be matched by name"):
        next(hd.kfold(2, stratify=stratify))


LABEL_VALUES = {
    "text": pd.Series(["a", "b", "nan", None, "a", "NA", " a", "2"]),
    "whole floats": pd.Series([1.0, 2.0, np.nan, 1.0]),
    "fractions": pd.Series([1.5, 2.0, np.nan]),
    "mixed objects": pd.Series([1, "1", 2, None, "none"], dtype=object),
    "categorical": pd.Series(pd.Categorical(["a", "nan", None, "2", "a"], categories=["2", "a", "nan", "unused"])),
    "categorical floats": pd.Series(pd.Categorical([1.0, 2.0, None, 2.0])),
    "string dtype": pd.Series(["a", "b", None, "n/a"], dtype="string"),
    "nullable integers": pd.Series([1, 2, None, 1], dtype="Int64"),
}


@pytest.mark.parametrize("values", LABEL_VALUES.values(), ids=LABEL_VALUES.keys())
def test_codes_for_matches_every_label_on_its_own(values):
    from hidata._core import _utils

    categories = pd.Index(["1", "2", "a", "b", "1.5", "nan"])
    labels = np.asarray(_utils.string_labels(values), dtype=object)
    expected = np.asarray(categories.get_indexer(labels), dtype=np.int64)
    expected[pd.isna(labels)] = -1
    np.testing.assert_array_equal(_utils.codes_for(values, categories), expected)
    missing = values.isna() | values.astype(str).str.strip().str.lower().isin(_utils.MISSING_LABELS)
    np.testing.assert_array_equal(_utils.is_missing_label(values), missing.to_numpy())
    codes, flagged = _utils.codes_and_missing(values, categories)
    np.testing.assert_array_equal(codes, expected)
    np.testing.assert_array_equal(flagged, missing.to_numpy())
