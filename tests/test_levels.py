import warnings

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

import hidata
from hidata import HiData, read_h5_hd, read_hd, read_zarr_hd, validate
from hidata._core import _utils

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def _open(path):
    return h5py.File(path) if str(path).endswith(".h5") else zarr.open(str(path), mode="r")


def test_from_anndata_with_subgroups_places_columns_at_their_level(hd3, patients):
    assert hd3.level_names == ("group", "subgroup", "cell")
    assert set(patients.columns) <= set(hd3.obs.columns)
    assert {"day", "tissue", "patient"} <= set(hd3.S.columns)
    assert not {"day", "tissue", "age"} & set(hd3.C.obs.columns)
    assert hd3.n_subgroups == hd3.S.shape[0]
    np.testing.assert_array_equal(hd3.n_subgroups_per_group.to_numpy(), [1 + (i % 3) for i in range(8)][: hd3.n_groups])
    assert np.all(np.diff(hd3.subgroup_group_codes) >= 0)
    assert np.all(np.diff(hd3.subgroup_codes) >= 0)
    np.testing.assert_array_equal(hd3.group_codes, hd3.subgroup_group_codes[hd3.subgroup_codes])
    validate(hd3)
    levels = hd3.infer_levels()
    assert levels.loc["day", "level"] == "subgroup"
    assert levels.loc["cell_type", "level"] == "cell"


def test_subgroups_with_mixed_groups_are_rejected(visits):
    visits.obs.loc[visits.obs_names[0], "subgroup"] = visits.obs["subgroup"].iloc[-1]
    if visits.obs["patient"].iloc[0] != visits.obs["patient"].iloc[-1]:
        with pytest.raises(ValueError, match="several groups"):
            HiData.from_anndata(visits, group_id="patient", subgroup_id="subgroup")


def test_filters_cascade_across_levels(hd3):
    later = hd3.filter_subgroups("day > 0")
    assert (later.S["day"] > 0).all()
    assert set(later.obs_names) == set(later.S["patient"])
    assert set(later.C.obs["subgroup"]) == set(later.S.index)
    validate(later)
    old = hd3.filter_subgroups("age > 50 and tissue == 'blood'")
    assert (old.obs["age"] > 50).all()
    groups = hd3.filter_groups(["P1", "P5"])
    assert set(groups.S["patient"]) == {"P1", "P5"}
    b_cells = hd3.filter_cells("cell_type == 'B'")
    assert (b_cells.n_cells_per_subgroup > 0).all()
    validate(b_cells)
    with pytest.raises(ValueError, match="no groups"):
        hd3.filter_subgroups("day > 1000")


def test_level_aware_operations(hd3, visits):
    fractions = hd3.composition("cell_type", level="subgroup")
    assert fractions.shape[0] == hd3.n_subgroups
    assert "cell_type_fractions" in hd3.levels["subgroup"].obsm
    hd3.aggregate(obsm="X_pca", key_added="pca", level="subgroup")
    expected = visits.obsm["X_pca"][(visits.obs["subgroup"] == hd3.S.index[0]).to_numpy()].mean(axis=0)
    np.testing.assert_allclose(hd3.levels["subgroup"].obsm["pca"][0], expected)
    bulk = hd3.pseudobulk(layer="counts", level="subgroup")
    assert bulk.n_obs == hd3.n_subgroups
    assert {"subgroup", "patient", "day", "age", "n_cells", "total_counts"} <= set(bulk.obs.columns)
    np.testing.assert_allclose(bulk.obs["total_counts"], np.asarray(bulk.X.sum(axis=1)).ravel())
    hd3.push_obs("age", to_level="subgroup")
    assert "age" in hd3.S.columns
    hd3.push_obs("day", level="subgroup")
    np.testing.assert_array_equal(hd3.C.obs["day"].to_numpy(), hd3.S["day"].to_numpy()[hd3.subgroup_codes])
    collected = hd3.pull_obs({"n_counts": "median_counts"}, func="median", level="subgroup")
    assert collected == ["median_counts"] and "median_counts" in hd3.S.columns
    capped = hd3.subsample_cells(3, level="subgroup")
    assert (capped.n_cells_per_subgroup <= 3).all()
    big = hd3.filter_groups_by_cells(10, level="subgroup")
    assert (big.n_cells_per_subgroup >= 10).all()
    first = hd3.S.index[0]
    assert hd3.cells_of(first, level="subgroup").obs["subgroup"].unique().tolist() == [first]


def test_nest_flatten_and_collapse(hd, hd3):
    hd.obs["site"] = hd.obs["site"].astype(str)
    hd.obsm["emb"] = np.arange(hd.n_groups * 2.0).reshape(-1, 2)
    nested = hd.nest("site")
    assert nested.level_names == ("group", "subgroup", "cell")
    assert nested.group_id == "site" and nested.subgroup_id == "patient"
    assert nested.obs_names.tolist() == ["A", "B"]
    positions = hd.obs_names.get_indexer(nested.levels["subgroup"].obs_names)
    np.testing.assert_array_equal(nested.levels["subgroup"].obsm["emb"], hd.obsm["emb"][positions])
    validate(nested)
    flat = nested.flatten()
    assert flat.level_names == ("group", "cell") and flat.group_id == "patient"
    assert "site" in flat.obs.columns and flat.n_groups == hd.n_groups
    validate(flat)
    sites = hd3.collapse("site")
    assert sites.level_names == ("group", "subgroup", "cell")
    assert set(sites.S["site"]) == {"A", "B"}
    validate(sites)


def test_get_df_moves_values_between_levels(hd3, visits):
    cells = hd3.get_df(["age", "day", "cell_type", "gene1"], level="cell")
    assert cells.index.equals(hd3.C.obs_names)
    np.testing.assert_array_equal(cells["age"].to_numpy(), hd3.obs["age"].to_numpy()[hd3.group_codes])
    np.testing.assert_array_equal(cells["day"].to_numpy(), hd3.S["day"].to_numpy()[hd3.subgroup_codes])
    groups = hd3.get_df(["age", "n_counts", "gene1", "X_pca"], level="group")
    expected = visits.obs.groupby("patient", observed=True)["n_counts"].mean().loc[hd3.obs_names]
    np.testing.assert_allclose(groups["n_counts"], expected)
    assert [c for c in groups.columns if c.startswith("X_pca")] == [f"X_pca_{i}" for i in range(4)]
    subgroups = hd3.get_df(["age", "day"], level="subgroup")
    np.testing.assert_array_equal(subgroups["age"].to_numpy(), hd3.obs["age"].to_numpy()[hd3.subgroup_group_codes])
    with pytest.raises(ValueError, match="composition"):
        hd3.get_df("cell_type", level="group")
    with pytest.raises(KeyError):
        hd3.get_df("nothing")
    hd3.C.obs["age"] = 1.0
    explicit = hd3.get_df(["G:age", "C:age"], level="cell")
    assert explicit["C:age"].eq(1.0).all() and not explicit["G:age"].eq(1.0).all()


def test_partial_modalities_coverage_and_complete_cases(cells, patients):
    blood = cells[cells.obs["patient"].isin(["P0", "P1", "P2", "P3"]).to_numpy()].copy()
    marrow = cells[cells.obs["patient"].isin(["P2", "P3", "P4"]).to_numpy()].copy()
    marrow.obs_names = [f"m_{name}" for name in marrow.obs_names]
    marrow.var_names = [f"peak{i}" for i in range(marrow.n_vars)]
    mdata = md.MuData({"blood": blood, "marrow": marrow})
    mdata.obs["patient"] = pd.concat([blood.obs["patient"], marrow.obs["patient"]]).loc[mdata.obs_names].astype(str)
    hd = HiData(G=patients, C=mdata, group_id="patient")
    coverage = hd.coverage()
    assert coverage.loc["P0", "marrow"] == 0 and coverage.loc["P4", "blood"] == 0
    assert coverage.loc["P2", "blood"] > 0 and coverage.loc["P2", "marrow"] > 0
    both = hd.complete_cases()
    assert both.obs_names.tolist() == ["P2", "P3"]
    hd.aggregate(mod="marrow", key_added="marrow_mean")
    assert np.isnan(hd.obsm["marrow_mean"].loc["P0"]).all()
    assert np.isfinite(hd.obsm["marrow_mean"].loc["P4"]).all()
    bulk = hd.pseudobulk(mod="blood")
    assert set(bulk.obs_names) == {"P0", "P1", "P2", "P3"}
    values = hd.get_df("marrow:peak0", level="group")
    assert values["marrow:peak0"].isna().sum() == 2


def test_composition_transforms_and_membership(hd):
    clr = hd.composition("cell_type", clr=True)
    np.testing.assert_allclose(clr.sum(axis=1), 0, atol=1e-10)
    rng = np.random.default_rng(0)
    membership = pd.DataFrame(
        rng.random((hd.n_cells, 5)) > 0.6, index=hd.C.obs_names, columns=[f"nh{i}" for i in range(5)]
    )
    hd.C.obsm["nhoods"] = membership.astype(float)
    counts = hd.composition("nhoods", normalize=False)
    expected = membership.groupby(hd.C.obs["patient"].to_numpy()).sum().loc[hd.obs_names]
    np.testing.assert_array_equal(counts.to_numpy(), expected.to_numpy())


def test_cell_weights(hd):
    weights = hd.cell_weights()
    per_group = weights.groupby(hd.C.obs["patient"].to_numpy()).sum()
    np.testing.assert_allclose(per_group.to_numpy(), per_group.iloc[0])
    np.testing.assert_allclose(weights.mean(), 1)
    balanced = hd.cell_weights(by="disease")
    per_class = balanced.groupby(hd.obs["disease"].to_numpy()[hd.group_codes]).sum()
    np.testing.assert_allclose(per_class.to_numpy(), per_class.iloc[0])


def test_grouped_splits_keep_groups_together(hd):
    train, test = hd.train_test_split(0.5, groups="site", random_state=0)
    assert set(train.obs["site"]).isdisjoint(test.obs["site"])
    for fold_train, fold_test in hd.kfold(2, groups="site"):
        assert set(fold_train.obs["site"]).isdisjoint(fold_test.obs["site"])


def test_splits_keep_subgroups_of_a_group_together(hd3):
    train, test = hd3.train_test_split(0.25, random_state=2)
    assert set(train.S["patient"]).isdisjoint(test.S["patient"])
    assert train.n_subgroups + test.n_subgroups == hd3.n_subgroups


def test_namespace_registration(hd):
    @hidata.register_hidata_namespace("toy")
    class Toy:
        def __init__(self, hd):
            self._hd = hd

        def largest(self):
            return self._hd.n_cells_per_group.idxmax()

    assert hd.toy.largest() == hd.n_cells_per_group.idxmax()
    assert hd.toy is hd.toy
    delattr(HiData, "toy")


def test_update_after_in_place_edit(hd):
    hd.C.obs["patient"] = hd.C.obs["patient"].replace({"P0": "P1"})
    hd.update()
    assert "P0" not in hd.obs_names
    validate(hd)


def test_group_column_in_cells_must_agree(visits):
    from hidata import HiData

    hd = HiData.from_anndata(visits, group_id="patient", subgroup_id="subgroup")
    cells = hd.C.copy()
    cells.obs["patient"] = cells.obs["patient"].astype(str)
    cells.obs.iloc[0, cells.obs.columns.get_loc("patient")] = "P7" if cells.obs["patient"].iloc[0] != "P7" else "P0"
    with pytest.raises(ValueError, match="disagrees"):
        hd.C = cells


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_roundtrip_three_levels(tmp_path, hd3, extension):
    hd3.composition("cell_type", level="subgroup")
    hd3.levels["subgroup"].obsp["dist"] = np.eye(hd3.n_subgroups)
    hd3.obsm["emb"] = np.ones((hd3.n_groups, 2))
    path = tmp_path / f"visits{extension}"
    hd3.write_hd(str(path))
    loaded = read_h5_hd(path) if extension.endswith("h5") else read_zarr_hd(path)
    assert loaded.level_names == hd3.level_names
    pd.testing.assert_frame_equal(loaded.S, hd3.S)
    pd.testing.assert_frame_equal(
        loaded.levels["subgroup"].obsm["cell_type_fractions"],
        hd3.levels["subgroup"].obsm["cell_type_fractions"],
        check_names=False,
    )
    np.testing.assert_array_equal(loaded.levels["subgroup"].obsp["dist"], np.eye(hd3.n_subgroups))
    np.testing.assert_array_equal(loaded.obsm["emb"], hd3.obsm["emb"])
    validate(loaded)


def test_concat_three_levels(hd3):
    first = hd3.filter_groups(["P0", "P1"])
    second = hd3.filter_groups(["P2"])
    first.composition("cell_type", level="subgroup")
    second.composition("cell_type", level="subgroup")
    merged = hidata.concat([first, second])
    assert merged.n_subgroups == first.n_subgroups + second.n_subgroups
    assert merged.levels["subgroup"].obsm["cell_type_fractions"].index.equals(merged.levels["subgroup"].obs_names)
    validate(merged)


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_ehrdata_group_level(tmp_path, cells, patients, extension, ehrdata):
    ed = ehrdata
    rng = np.random.default_rng(0)
    series = rng.normal(size=(len(patients), 3, 4))
    edata = ed.EHRData(
        X=series.mean(axis=2),
        obs=patients.copy(),
        var=pd.DataFrame(index=["sbp", "dbp", "crp"]),
        layers={"tem_data": series},
        tem=pd.DataFrame({"day": [0, 7, 14, 28]}, index=[f"t{i}" for i in range(4)]),
    )
    hd = HiData(G=edata, C=cells, group_id="patient")
    sub = hd.filter_groups(["P1", "P3"])
    assert type(sub.G).__name__ == "EHRData"
    assert sub.G.layers["tem_data"].shape == (2, 3, 4)
    assert "tem" in repr(hd)
    path = tmp_path / f"ehr{extension}"
    hd.write_hd(str(path))
    store = _open(path)
    assert set(store.keys()) == {"G", "C"}
    h5 = extension == ".hd.h5"
    assert store["G"].attrs["ehrdata-encoding-type" if h5 else "encoding-type"] == "ehrdata"
    np.testing.assert_allclose(ed.io.read_elem(store["G"]).layers["tem_data"], series)
    plain = read_elem(store["G"] if h5 else store["G"]["anndata"])
    assert type(plain).__name__ == "AnnData" and plain.n_obs == len(patients)
    if h5:
        store.close()
    loaded = read_hd(path)
    assert type(loaded.G).__name__ == "EHRData"
    np.testing.assert_allclose(loaded.G.layers["tem_data"], hd.G.layers["tem_data"])
    assert loaded.G.tem["day"].tolist() == [0, 7, 14, 28]
    crp = hd.get_df("crp", level="cell")
    np.testing.assert_allclose(crp["crp"].to_numpy(), hd.G.X[:, 2][hd.group_codes])


def test_group_level_is_built_from_the_subgroup_table(hd3):
    subgroups = hd3.S.copy()
    built = HiData(S=subgroups, C=hd3.C, group_id=hd3.group_id, subgroup_id=hd3.subgroup_id)
    assert list(built.G.index) == list(hd3.G.index)
    assert built.G.shape[1] == 0
    assert built.n_subgroups == hd3.n_subgroups and built.n_cells == hd3.n_cells
    np.testing.assert_array_equal(built.group_codes, hd3.group_codes)
    with pytest.raises(TypeError, match="subgroup table S"):
        HiData(C=hd3.C, group_id=hd3.group_id)


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_ehrdata_subgroup_level(tmp_path, hd3, extension, ehrdata):
    ed = ehrdata
    rng = np.random.default_rng(1)
    subgroups = hd3.S.copy()
    series = rng.normal(size=(len(subgroups), 2, 3))
    edata = ed.EHRData(
        X=series.mean(axis=2),
        obs=subgroups,
        var=pd.DataFrame(index=["crp", "il6"]),
        layers={"tem_data": series},
        tem=pd.DataFrame({"hour": [0, 12, 24]}, index=["h0", "h12", "h24"]),
    )
    hd = HiData(G=hd3.G, S=edata, C=hd3.C, group_id=hd3.group_id, subgroup_id=hd3.subgroup_id)
    assert type(hd.S).__name__ == "EHRData"
    sub = hd.filter_groups(hd.obs_names[:2])
    assert sub.S.layers["tem_data"].shape == (sub.n_subgroups, 2, 3)
    path = tmp_path / f"ehr_subgroups{extension}"
    sub.write_hd(str(path))
    loaded = read_hd(path)
    assert type(loaded.S).__name__ == "EHRData"
    np.testing.assert_allclose(loaded.S.layers["tem_data"], sub.S.layers["tem_data"])


def test_group_sum_of_a_sparse_matrix_matches_dense_without_deprecations():
    dense = np.arange(12.0).reshape(4, 3)
    codes = np.array([0, 1, 0, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy 2 deprecates building a csr_matrix
        matrix = sp.csr_matrix(dense)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for transform in (None, np.log1p):
            expected = _utils.group_sum(dense, codes, 2, transform)
            np.testing.assert_allclose(_utils.group_sum(matrix, codes, 2, transform), expected)
            np.testing.assert_allclose(_utils.group_sum(sp.csr_array(dense), codes, 2, transform), expected)


def _rich_ehrdata(ed, patients):
    """EHRData with a 3D X, a sparse COO layer, datetimes, object columns with None and a numeric obsm frame."""
    import sparse

    n = len(patients)
    series = np.random.default_rng(4).normal(size=(n, 3, 4))
    obs = patients.assign(
        visit=pd.date_range("2024-01-01", periods=n, freq="D"),
        count=np.array([1, None] * (n // 2), dtype=object),
        note=np.array(["a", None] * (n // 2), dtype=object),
    )
    return ed.EHRData(
        X=series,
        obs=obs,
        var=pd.DataFrame(index=["sbp", "dbp", "crp"]),
        layers={"coo": sparse.COO.from_numpy(np.where(series > 1, series, np.nan), fill_value=np.nan)},
        obsm={"profile": pd.DataFrame(series[:, :, 0], index=obs.index, columns=["g1", "g2", "g3"])},
        tem=pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="h")}, index=[f"t{i}" for i in range(4)]),
    )


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_ehrdata_level_is_stored_as_ehrdata_stores_it(tmp_path, cells, patients, extension, ehrdata):
    sparse = pytest.importorskip("sparse")
    edata = _rich_ehrdata(ehrdata, patients)
    path = tmp_path / f"rich{extension}"
    HiData(G=edata, C=cells, group_id="patient").write_hd(str(path))
    store = _open(path)
    slots = store["G"] if extension == ".hd.h5" else store["G"]["anndata"]
    assert "profile" in slots["obsm_columns"]
    # packed into one array
    assert slots["obsm"]["profile"].attrs["encoding-type"] == "array"
    if extension == ".hd.h5":
        store.close()
    loaded = read_hd(path)
    G = loaded.G
    np.testing.assert_allclose(G.X, edata.X)
    assert isinstance(G.layers["coo"], sparse.COO)
    np.testing.assert_allclose(G.layers["coo"].todense(), edata.layers["coo"].todense())
    assert G.obs["visit"].iloc[1] == "2024-01-02T00:00:00" and G.tem["time"].iloc[1] == "2024-01-01T01:00:00"
    assert G.obs["count"].dtype == "Int64" and G.obs["count"].isna().sum() == len(patients) // 2
    assert G.obs["note"].iloc[0] == "a" and G.obs["note"].isna().sum() == len(patients) // 2
    pd.testing.assert_frame_equal(G.obsm["profile"], edata.obsm["profile"], check_names=False)
    if extension == ".hd.zarr":
        assert isinstance(loaded.C.X, ad.abc.CSRDataset)
    validate(loaded)


def test_x_chunks_for_an_ehrdata_level_warn(tmp_path, cells, patients, ehrdata):
    hd = HiData(G=_rich_ehrdata(ehrdata, patients), C=cells, group_id="patient")
    with pytest.warns(UserWarning, match="x_chunks does not apply to the EHRData level G"):
        hd.write_zarr_hd(str(tmp_path / "chunks.hd.zarr"), x_chunks={"G": (2, 2), "C": (8, 4)})
    np.testing.assert_allclose(read_zarr_hd(tmp_path / "chunks.hd.zarr").G.X, hd.G.X)
    # a shared shape is meant for C
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="x_chunks", category=UserWarning)
        hd.write_zarr_hd(str(tmp_path / "shared.hd.zarr"), x_chunks=(8, 4))


def test_ehrdata_cell_level_is_rejected(cells, patients, ehrdata):
    edata_cells = ehrdata.EHRData(X=cells.X, obs=cells.obs, var=cells.var)
    with pytest.raises(TypeError, match="an EHRData belongs to G or S"):
        HiData(G=patients, C=edata_cells, group_id="patient")


def test_ehrdata_level_without_stamp_is_rejected(tmp_path, cells, patients, ehrdata):
    path = tmp_path / "dev.hd.h5"
    HiData(G=_rich_ehrdata(ehrdata, patients), C=cells, group_id="patient").write_hd(str(path))
    with h5py.File(path, "a") as f:
        del f["G"].attrs["ehrdata-encoding-type"]
    with pytest.raises(ValueError, match="layout of a development version"):
        read_hd(path)
