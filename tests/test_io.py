import gc

import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata.io import read_elem

from hidata import HiData, read_h5_hd, read_hd, read_zarr_hd, validate

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def _decorate(hd):
    hd.obsm["emb"] = pd.DataFrame(np.arange(hd.n_groups * 2.0).reshape(-1, 2), index=hd.obs_names, columns=["a", "b"])
    hd.obsm["arr"] = np.ones((hd.n_groups, 3))
    hd.obsp["dist"] = np.eye(hd.n_groups)
    hd.aggregate(key_added="pb", sync_var=True)
    hd.uns["study"] = "test"
    return hd


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_roundtrip_table_G(tmp_path, hd, extension):
    hd = _decorate(hd)
    path = tmp_path / f"cohort{extension}"
    hd.write_hd(str(path))
    loaded = read_hd(path)
    assert loaded.G_type == "table"
    assert loaded.group_id == "patient"
    assert loaded.shape == hd.shape
    pd.testing.assert_frame_equal(loaded.obs, hd.obs)
    pd.testing.assert_frame_equal(loaded.obsm["emb"], hd.obsm["emb"])
    np.testing.assert_array_equal(loaded.obsp["dist"], hd.obsp["dist"])
    assert loaded._var_dims_to_sync == ["pb"]
    assert loaded.uns["study"] == "test"
    np.testing.assert_array_equal(np.asarray(loaded.C.obs_names), np.asarray(hd.C.obs_names))
    validate(loaded)


def test_roundtrip_anndata_G(tmp_path, hd_genotypes):
    hd = _decorate(hd_genotypes)
    path = tmp_path / "cohort.hd.h5"
    hd.write_h5_hd(path)
    loaded = read_h5_hd(path)
    assert loaded.G_type == "anndata"
    np.testing.assert_array_equal(loaded.G.X, hd.G.X)
    assert set(loaded.obsm) == set(hd.obsm)


def test_zarr_reads_dense_X_lazily(tmp_path, cells, genotypes):
    cells.X = cells.X.toarray()
    hd = HiData(G=genotypes, C=cells, group_id="patient")
    path = tmp_path / "dense.hd.zarr"
    hd.write_zarr_hd(path, x_chunks={"G": (4, 2), "C": (16, 4)})
    loaded = read_zarr_hd(path)
    assert type(loaded.C.X).__module__.startswith("dask")
    np.testing.assert_allclose(loaded.C.X.compute(), hd.C.X)
    subset = loaded.filter_groups(["P1"])
    assert subset.C.X.shape[0] == hd.n_cells_per_group["P1"]


def test_zarr_keeps_sparse_X_on_disk_and_aggregates_it_by_block(tmp_path, monkeypatch, hd_genotypes):
    from anndata.abc import CSRDataset

    from hidata._core import _utils

    path = tmp_path / "sparse.hd.zarr"
    hd_genotypes.write_zarr_hd(path)
    loaded = read_zarr_hd(path)
    assert isinstance(loaded.C.X, CSRDataset)
    monkeypatch.setattr(_utils, "BLOCK_NNZ", 25)
    monkeypatch.setattr(type(loaded.C.X), "to_memory", lambda self: pytest.fail("the whole matrix was read"))
    selections = [
        lambda d: d,
        lambda d: d.filter_groups(["P1", "P4", "P6"]),
        lambda d: d.sel(C_var=["gene3", "gene0", "gene7"]),
    ]
    for select in selections:
        for func in ("sum", "mean", "var", "count_nonzero"):
            for cell_filter in ({}, {"filter_key": "cell_type", "filter_value": "B"}):
                expected, result = select(hd_genotypes), select(loaded)
                expected.aggregate(key_added="agg", func=func, **cell_filter)
                result.aggregate(key_added="agg", func=func, **cell_filter)
                assert result.obsm["agg"].columns.equals(expected.obsm["agg"].columns)
                np.testing.assert_allclose(result.obsm["agg"].to_numpy(), expected.obsm["agg"].to_numpy())


def test_zarr_reads_selected_G_var_columns(tmp_path, hd_genotypes):
    path = tmp_path / "genotypes.hd.zarr"
    hd_genotypes.write_zarr_hd(path)
    loaded = read_zarr_hd(path, var_columns=["chrom", "pos"])
    assert list(loaded.G.var.columns) == ["chrom", "pos"]
    assert loaded.G.var_names.equals(hd_genotypes.G.var_names)
    np.testing.assert_array_equal(loaded.G.var["pos"], hd_genotypes.G.var["pos"])


def test_roundtrip_mudata_C_h5(tmp_path, cells, patients):
    protein = cells[:, :3].copy()
    protein.var_names = ["CD3", "CD19", "CD56"]
    mdata = md.MuData({"rna": cells.copy(), "prot": protein})
    mdata.obs["patient"] = cells.obs["patient"].to_numpy()
    hd = HiData(G=patients, C=mdata, group_id="patient")
    path = tmp_path / "multi.hd.h5"
    hd.write_h5_hd(path)
    loaded = read_h5_hd(path)
    assert isinstance(loaded.C, md.MuData)
    assert loaded.n_cells == hd.n_cells
    assert list(loaded.C.obs.columns) == list(hd.C.obs.columns)
    assert list(loaded.C.mod) == ["rna", "prot"]


def test_roundtrip_mudata_zarr_without_pulling_columns(tmp_path, cells, patients):
    protein = cells[:, :3].copy()
    protein.var_names = ["CD3", "CD19", "CD56"]
    mdata = md.MuData({"rna": cells.copy(), "prot": protein})
    mdata.obs = mdata.obs[[]].assign(patient=cells.obs["patient"].to_numpy())
    hd = HiData(G=patients, C=mdata, group_id="patient")
    hd.write_zarr_hd(tmp_path / "multi.hd.zarr")
    loaded = read_zarr_hd(tmp_path / "multi.hd.zarr")
    assert isinstance(loaded.C, md.MuData)
    assert list(loaded.C.obs.columns) == ["patient"]
    np.testing.assert_allclose(loaded.C.mod["prot"].X.to_memory().toarray(), hd.C.mod["prot"].X.toarray())
    assert list(hd.C.obs.columns) == ["patient"]


@pytest.fixture
def donordata():
    return pytest.importorskip("donordata")


def _same(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_names=False, check_frame_type=False)
    else:
        _matrices_equal(actual, expected)


def _assert_same_levels(loaded, written):
    assert (loaded.group_id, loaded.subgroup_id) == (written.donor_id, written.sample_id)
    assert loaded.G_type == written.D_type
    assert loaded._var_dims_to_sync == written._var_dims_to_sync
    assert loaded.uns == written.uns
    for new, old in ((loaded._G, written._D), (loaded._S, written._S), (loaded._C, written._C)):
        assert (new is None) == (old is None)
        if old is None:
            continue
        pd.testing.assert_frame_equal(new.obs, old.obs, check_frame_type=False)
        pd.testing.assert_frame_equal(new.var, old.var, check_frame_type=False)
        if old.n_vars:
            _matrices_equal(new.X, old.X)
        assert set(new.obsm) == set(old.obsm) and set(new.obsp) == set(old.obsp)
        for key in old.obsm:
            _same(new.obsm[key], old.obsm[key])
        for key in old.obsp:
            _same(new.obsp[key], old.obsp[key])
    validate(loaded)


def _donordata_cohort(donordata, kind, cells, visits, genotypes, patients):
    if kind == "table":
        written = donordata.DonorData.from_anndata(cells, donor_id="patient")
    elif kind == "anndata":
        cells = cells.copy()
        cells.obs = cells.obs.drop(columns=patients.columns)
        genotypes.obs = genotypes.obs.join(patients)
        written = donordata.DonorData(D=genotypes, C=cells, donor_id="patient")
    else:
        written = donordata.DonorData.from_anndata(visits, donor_id="patient", sample_id="subgroup")
        written.aggregate(key_added="profiles", level="sample")
        written.composition("cell_type", level="sample", normalize=False)
    written.obsm["emb"] = pd.DataFrame(
        np.arange(written.n_donors * 2.0).reshape(-1, 2), index=written.obs_names, columns=["a", "b"]
    )
    written.obsm["arr"] = np.ones((written.n_donors, 3))
    written.obsp["dist"] = np.eye(written.n_donors)
    written.aggregate(key_added="pb", sync_var=True)
    written.uns["study"] = "test"
    return written


@pytest.mark.parametrize("kind", ["table", "anndata", "subgroups"])
@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_reads_stores_written_by_donordata(tmp_path, donordata, cells, visits, genotypes, patients, kind, extension):
    import zarr

    written = _donordata_cohort(donordata, kind, cells, visits, genotypes, patients)
    path = tmp_path / f"cohort{extension}"
    written.write_dd(str(path))
    loaded = read_hd(path, lazy=False)
    _assert_same_levels(loaded, written)
    assert loaded.n_groups == written.n_donors and loaded.n_subgroups == written.n_samples

    again = tmp_path / f"cohort{extension.replace('.dd', '.hd')}"
    loaded.write_hd(again)
    is_h5 = extension.endswith(".h5")
    store = h5py.File(again, "r") if is_h5 else zarr.open_group(str(again), mode="r")
    assert store.attrs["encoding-type"] == "hidata" and store.attrs["encoding-version"] == "0.1.0"
    assert set(store.keys()) - {"uns"} == ({"G", "S", "C"} if kind == "subgroups" else {"G", "C"})
    assert store.attrs["group_id"] == "patient" and "donor_id" not in store.attrs
    if kind != "anndata":
        assert store["G"].attrs["hidata-table"] and "donordata-table" not in store["G"].attrs
    if is_h5:
        store.close()
    _assert_same_levels(read_hd(again, lazy=False), written)


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_reads_stores_with_the_G_group_of_cellink(tmp_path, donordata, cells, extension):
    import zarr

    written = donordata.DonorData.from_anndata(cells, donor_id="patient")
    path = tmp_path / f"old{extension}"
    written.write_dd(str(path))
    if extension.endswith(".h5"):
        with h5py.File(path, "r+") as f:
            f.move("D", "G")
            f.attrs["encoding-version"] = "0.2.0"
    else:
        (path / "D").rename(path / "G")
        zarr.open_group(str(path), mode="r+").attrs["encoding-version"] = "0.2.0"
    with pytest.warns(UserWarning, match="layout of donordata files of version 0.2.0"):
        loaded = read_hd(path, lazy=False)
    _assert_same_levels(loaded, written)


def test_donordata_encoding_checks(tmp_path, donordata, cells):
    path = tmp_path / "cohort.dd.h5"
    donordata.DonorData.from_anndata(cells, donor_id="patient").write_h5_dd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-version"] = "9.0.0"
    with pytest.warns(UserWarning, match="encoding-version='9.0.0'"):
        read_h5_hd(path)


def test_encoding_checks(tmp_path, hd):
    path = tmp_path / "cohort.hd.h5"
    hd.write_h5_hd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-version"] = "0.1.0"
    read_h5_hd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-version"] = "9.0.0"
    with pytest.warns(UserWarning, match="encoding-version"):
        read_h5_hd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-type"] = "anndata"
    with pytest.raises(ValueError, match="encoding-type"):
        read_h5_hd(path)


def test_format_detection(tmp_path, hd):
    with pytest.raises(ValueError, match="extension"):
        hd.write_hd(str(tmp_path / "cohort.txt"))
    hd.write_hd(str(tmp_path / "cohort"), fmt="h5")
    assert (tmp_path / "cohort.hd.h5").exists()
    with pytest.raises(ValueError, match="extension"):
        read_hd(tmp_path / "cohort")


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_mudata_with_shared_variable_names_and_named_indices(tmp_path, extension):
    from anndata import AnnData

    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(20)]
    rna = AnnData(
        rng.poisson(2, (20, 6)).astype(np.float32),
        obs=pd.DataFrame({"n_counts": rng.random(20)}, index=pd.Index(cells, name="index")),
        var=pd.DataFrame(index=pd.Index(["CD86", "g1", "g2", "g3", "g4", "g5"], name="index")),
    )
    adt = AnnData(
        rng.random((20, 3)).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(cells, name="index")),
        var=pd.DataFrame(index=pd.Index(["CD86", "PDL1", "p3"], name="index")),
    )
    mdata = md.MuData({"rna": rna, "adt": adt})
    mdata.obs = pd.DataFrame(
        {"guide": np.repeat(["g1", "g2", "g3", "g4"], 5), "gene": np.repeat(["A", "A", "B", "NT"], 5)},
        index=pd.Index(cells),
    )

    hd = HiData.from_anndata(mdata, group_id="gene", subgroup_id="guide")
    assert (hd.n_groups, hd.n_subgroups) == (3, 4)
    assert list(hd.C.obs.columns) == ["guide", "gene"]
    assert list(hd.copy().C.obs.columns) == ["guide", "gene"]
    path = tmp_path / f"screen{extension}"
    hd.write_hd(str(path))
    loaded = read_hd(path)
    assert list(loaded.C.obs.columns) == ["guide", "gene"]
    np.testing.assert_allclose(loaded.C.mod["rna"].X, hd.C.mod["rna"].X)
    np.testing.assert_array_equal(np.asarray(loaded.C.obsmap["adt"]), np.asarray(hd.C.obsmap["adt"]))
    assert list(loaded.C.obs_names) == list(hd.C.obs_names)
    validate(loaded)
    subset = loaded.filter_groups(["A"]).copy()
    assert subset.C.mod["adt"].n_obs == 10


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_wide_frames_in_obsm_are_written_as_one_array(tmp_path, hd3, extension):
    hd3.aggregate(key_added="profiles", level="subgroup")
    hd3.composition("cell_type", level="subgroup", normalize=False)
    hd3.obsm["mixed"] = pd.DataFrame({"a": np.arange(hd3.n_groups), "b": ["x"] * hd3.n_groups}, index=hd3.obs_names)
    path = tmp_path / f"cohort{extension}"
    hd3.write_hd(str(path))
    if extension == ".hd.h5":
        with h5py.File(path) as f:
            assert isinstance(f["S/obsm/profiles"], h5py.Dataset)
            assert isinstance(f["G/obsm/mixed"], h5py.Group)
            assert set(f.keys()) == {"G", "S", "C"}
            plain = read_elem(f["S"])
            assert plain.obsm["profiles"].shape == hd3.levels["subgroup"].obsm["profiles"].shape
    loaded = read_hd(path)
    subgroups = loaded.levels["subgroup"]
    for key in ("profiles", "cell_type_counts"):
        pd.testing.assert_frame_equal(subgroups.obsm[key], hd3.levels["subgroup"].obsm[key], check_names=False)
    pd.testing.assert_frame_equal(loaded.obsm["mixed"], hd3.obsm["mixed"], check_names=False)


def _matrices_equal(actual, expected):
    actual = actual.compute() if hasattr(actual, "compute") else actual
    actual = actual.to_memory() if hasattr(actual, "to_memory") else actual
    actual = actual.toarray() if hasattr(actual, "toarray") else np.asarray(actual)
    expected = expected.toarray() if hasattr(expected, "toarray") else np.asarray(expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dense_cells", [False, True], ids=["sparse C", "dense C"])
def test_h5_reads_matrices_lazily_and_gives_the_same_values(tmp_path, cells, genotypes, patients, dense_cells):
    from anndata.abc import CSRDataset

    cells = cells.copy()
    cells.obs = cells.obs.drop(columns=patients.columns)
    if dense_cells:
        cells.X = cells.X.toarray()
    hd = HiData(G=genotypes, C=cells, group_id="patient")
    path = tmp_path / "cohort.hd.h5"
    hd.write_h5_hd(path)
    eager, lazy = read_h5_hd(path), read_h5_hd(path, lazy=True)
    assert type(lazy.G.X).__module__.startswith("dask")
    assert type(lazy.C.X).__module__.startswith("dask") if dense_cells else isinstance(lazy.C.X, CSRDataset), type(
        lazy.C.X
    )
    _matrices_equal(lazy.G.X, eager.G.X)
    _matrices_equal(lazy.C.X, eager.C.X)
    subset = lazy.filter_groups(["P1", "P3"])
    _matrices_equal(subset.C.X, eager.filter_groups(["P1", "P3"]).C.X)
    lazy.aggregate(key_added="mean")
    eager.aggregate(key_added="mean")
    np.testing.assert_allclose(np.asarray(lazy.obsm["mean"]), np.asarray(eager.obsm["mean"]))
    validate(lazy)


def test_h5_reads_selected_G_var_columns(tmp_path, hd_genotypes):
    path = tmp_path / "cohort.hd.h5"
    hd_genotypes.write_h5_hd(path)
    for lazy in (False, True):
        assert list(read_h5_hd(path, lazy=lazy, var_columns=["chrom", "pos"]).G.var.columns) == ["chrom", "pos"]


def test_zarr_reads_into_memory_without_lazy(tmp_path, hd_genotypes):
    import scipy.sparse as sp

    path = tmp_path / "cohort.hd.zarr"
    hd_genotypes.write_zarr_hd(path)
    loaded = read_zarr_hd(path, lazy=False)
    assert isinstance(loaded.G.X, np.ndarray)
    assert sp.issparse(loaded.C.X)
    _matrices_equal(loaded.C.X, hd_genotypes.C.X)
    assert isinstance(read_hd(path, lazy=False).G.X, np.ndarray)
    assert type(read_hd(path).G.X).__module__.startswith("dask")


@pytest.mark.parametrize("extension", [".hd.zarr", ".hd.h5"])
def test_copy_of_a_lazily_read_object_is_independent(tmp_path, hd_genotypes, extension):
    path = tmp_path / f"cohort{extension}"
    hd_genotypes.write_hd(path)
    lazy = read_hd(path, lazy=True)
    copied = lazy.copy()
    _matrices_equal(copied.C.X, hd_genotypes.C.X)
    copied.C.obs["added"] = 1
    assert "added" not in lazy.C.obs.columns
    validate(copied)


def test_copy_of_a_lazily_read_mudata_level(tmp_path, cells, patients):
    import scipy.sparse as sp

    cells = cells.copy()
    cells.obs = cells.obs.drop(columns=patients.columns.drop("sex", errors="ignore"), errors="ignore")
    other = cells.copy()
    other.var_names = [f"p{i}" for i in range(other.n_vars)]
    mdata = md.MuData({"rna": cells, "prot": other})
    mdata.obs["patient"] = cells.obs["patient"].to_numpy()
    hd = HiData(G=patients, C=mdata, group_id="patient")
    path = tmp_path / "cohort.hd.zarr"
    hd.write_zarr_hd(path)
    copied = read_zarr_hd(path).copy()
    assert sp.issparse(copied.C.mod["rna"].X)
    validate(copied)


def _small_hd(X=None, layers=None):
    from anndata import AnnData

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2) if X is None else X
    obs = pd.DataFrame({"group": ["A"] * 3 + ["B"] * 3}, index=[f"c{i}" for i in range(6)])
    cells = AnnData(X=X, obs=obs, layers=layers or {})
    groups = pd.DataFrame({"age": [30, 40]}, index=pd.Index(["A", "B"], name="group"))
    return HiData(G=groups, C=cells, group_id="group")


@pytest.mark.parametrize("sparse_X", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("extension", [".hd.zarr", ".hd.h5"])
def test_writing_into_the_file_a_lazy_object_reads_from_raises(tmp_path, sparse_X, extension):
    import scipy.sparse as sp

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    hd = _small_hd(sp.csr_matrix(X) if sparse_X else X)
    path = tmp_path / f"cohort{extension}"
    hd.write_hd(path)
    lazy = read_hd(path, lazy=True)
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_hd(path)
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.filter_groups(["A"]).write_hd(path)
    _matrices_equal(read_hd(path, lazy=False).C.X, X)
    del lazy
    gc.collect()
    read_hd(path, lazy=False).write_hd(path)
    _matrices_equal(read_hd(path, lazy=False).C.X, X)


def test_writing_into_the_store_a_zarr_backed_matrix_reads_from_raises(tmp_path):
    import zarr

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    path = tmp_path / "cohort.hd.zarr"
    _small_hd(X).write_hd(path)
    lazy = read_hd(path, lazy=True)
    # readers such as cellink back X with the zarr.Array itself instead of a dask array
    lazy.C.X = zarr.open_group(str(path), mode="r")["C"]["X"]
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_hd(path)
    lazy.C.X = zarr.open_array(str(path / "C" / "X"), mode="r")
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_hd(path)
    _matrices_equal(read_hd(path, lazy=False).C.X, X)


def test_writing_into_the_file_an_h5py_backed_matrix_reads_from_raises(tmp_path):
    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    path = tmp_path / "cohort.hd.h5"
    _small_hd(X).write_hd(path)
    before = path.read_bytes()
    lazy = read_hd(path, lazy=True)
    with h5py.File(path, "r") as f:
        lazy.C.X = f["C"]["X"]
        with pytest.raises(ValueError, match="reads its matrices from"):
            lazy.write_hd(path)
    del lazy
    gc.collect()
    assert path.read_bytes() == before
    _matrices_equal(read_hd(path, lazy=False).C.X, X)


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
def test_nullable_columns_of_group_matrices_keep_their_type(tmp_path, extension):
    hd = _small_hd()
    hd.obsm["n"] = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")}, index=hd.obs_names)
    hd.obsm["flag"] = pd.DataFrame({"flag": pd.array([True, None], dtype="boolean")}, index=hd.obs_names)
    hd.write_hd(tmp_path / f"nullable{extension}")
    loaded = read_hd(tmp_path / f"nullable{extension}", lazy=False)
    assert str(loaded.obsm["n"]["n"].dtype) == "Int64"
    assert str(loaded.obsm["flag"]["flag"].dtype) == "boolean"


def test_zarr_keeps_raw_and_writes_float16(tmp_path):
    hd = _small_hd(np.ones((6, 2), dtype=np.float16))
    hd.C.raw = hd.C.copy()
    hd.write_zarr_hd(tmp_path / "cohort.hd.zarr")
    loaded = read_zarr_hd(tmp_path / "cohort.hd.zarr", lazy=False)
    assert loaded.C.raw is not None
    assert loaded.C.X.dtype == np.float16


def test_a_selection_of_a_lazy_object_with_dense_layers_can_be_written(tmp_path):
    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    _small_hd(X, layers={"counts": X * 2}).write_zarr_hd(tmp_path / "lazy.hd.zarr")
    subset = read_zarr_hd(tmp_path / "lazy.hd.zarr").filter_groups(["A"])
    subset.write_zarr_hd(tmp_path / "subset.hd.zarr")
    loaded = read_zarr_hd(tmp_path / "subset.hd.zarr", lazy=False)
    np.testing.assert_array_equal(loaded.C.layers["counts"], X[:3] * 2)


def test_written_mudata_maps_cells_to_modalities_by_name(tmp_path):
    from anndata import AnnData

    obs = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=["c0", "c1", "c2", "c3"])
    mdata = md.MuData({"rna": AnnData(X=np.arange(4.0).reshape(4, 1), obs=obs.copy())})
    mdata.obs["group"] = obs["group"]
    hd = HiData(G=pd.DataFrame(index=pd.Index(["B", "A"], name="group")), C=mdata, group_id="group")
    hd.write_h5_hd(tmp_path / "mdata.hd.h5")
    with h5py.File(tmp_path / "mdata.hd.h5", "r") as f:
        names = read_elem(f["C/obs"]).index
        mod_names = read_elem(f["C/mod/rna/obs"]).index
        obsmap = read_elem(f["C/obsmap/rna"])
    assert list(mod_names[obsmap - 1]) == list(names)


def test_read_hd_accepts_a_store_path_with_a_trailing_slash(tmp_path):
    _small_hd().write_zarr_hd(tmp_path / "cohort.hd.zarr")
    assert read_hd(f"{tmp_path / 'cohort.hd.zarr'}/").n_groups == 2
