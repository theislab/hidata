import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from hidata import HiData, HiDataSchemaError, read_hd, validate

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def test_from_anndata_moves_constant_columns(cells, hd, patients):
    assert hd.G_type == "table"
    assert isinstance(hd.G, pd.DataFrame)
    assert set(hd.obs.columns) == set(patients.columns)
    assert not set(patients.columns) & set(hd.C.obs.columns)
    assert {"cell_type", "n_counts", "patient"} <= set(hd.C.obs.columns)
    pd.testing.assert_frame_equal(hd.obs.loc[patients.index, patients.columns], patients, check_names=False)
    assert hd.obs["sex"].dtype == patients["sex"].dtype
    assert set(patients.columns) <= set(cells.obs.columns)
    assert hd.C.X is not None
    validate(hd)


def test_from_anndata_keep_in_cells_and_explicit_columns(cells):
    hd = HiData.from_anndata(cells, group_id="patient", columns=["age"], keep_in_cells=True)
    assert list(hd.obs.columns) == ["age"]
    assert "age" in hd.C.obs.columns
    with pytest.raises(ValueError, match="vary within groups"):
        HiData.from_anndata(cells, group_id="patient", columns=["cell_type"])


def test_cells_grouped_by_group_in_G_order(hd, cells):
    codes = hd.group_codes
    assert np.all(np.diff(codes) >= 0)
    for group in hd.obs_names:
        original = cells.obs_names[cells.obs["patient"] == group]
        assert hd.cells_of(group).obs_names.tolist() == original.tolist()
    offsets = hd.cell_offsets
    assert offsets[-1] == hd.n_cells
    np.testing.assert_array_equal(np.diff(offsets), hd.n_cells_per_group.to_numpy())


def test_table_with_group_column_and_unnamed_index(cells, patients):
    with_column = patients.reset_index()
    hd = HiData(G=with_column, C=cells, group_id="patient")
    assert hd.obs.index.name == "patient"
    assert hd.n_groups == len(patients)
    unnamed = patients.rename_axis(None)
    assert HiData(G=unnamed, C=cells, group_id="patient").n_groups == len(patients)


def test_integer_group_identifiers(cells, patients):
    table = patients.reset_index(drop=True)
    cells = cells.copy()
    cells.obs["patient"] = cells.obs["patient"].str[1:].astype(int)
    hd = HiData(G=table, C=cells, group_id="patient")
    assert hd.n_groups == len(table)
    assert hd.n_cells == cells.n_obs


def test_unmatched_groups_are_dropped(cells, patients):
    table = pd.concat([patients, patients.iloc[:1].rename(index={"P0": "P99"})])
    hd = HiData(G=table, C=cells[cells.obs["patient"] != "P1"], group_id="patient")
    assert "P99" not in hd.obs_names
    assert "P1" not in hd.obs_names
    assert hd.n_groups == len(patients) - 1


def test_errors(cells, patients):
    with pytest.raises(ValueError, match="No overlapping groups"):
        HiData(G=patients.rename(index=lambda x: x + "x"), C=cells, group_id="patient")
    with pytest.raises(ValueError, match="unique"):
        HiData(G=pd.concat([patients, patients]), C=cells, group_id="patient")
    with pytest.raises(ValueError, match="not found in C.obs"):
        HiData(G=patients, C=cells, group_id="nope")
    with pytest.raises(TypeError):
        HiData(G=patients.to_numpy(), C=cells, group_id="patient")


def test_anndata_G(hd_genotypes, genotypes):
    assert hd_genotypes.G_type == "anndata"
    assert hd_genotypes.G is genotypes
    hd_genotypes.obsm["emb"] = np.zeros((hd_genotypes.n_groups, 2))
    assert "emb" in genotypes.obsm
    assert hd_genotypes.shape == (8, 5, hd_genotypes.n_cells, 12)
    sub = hd_genotypes["P2", "snp1"]
    assert sub.shape[:2] == (1, 1)
    assert set(sub.C.obs["patient"]) == {"P2"}


def test_mudata_C(cells, patients):
    mdata = md.MuData({"rna": cells, "prot": cells[:, :3].copy()})
    mdata.obs["patient"] = cells.obs["patient"]
    hd = HiData(G=patients, C=mdata, group_id="patient")
    assert hd.n_cells == cells.n_obs
    sub = hd.filter_groups(["P0", "P1"])
    assert set(sub.C.obs["patient"]) == {"P0", "P1"}
    pseudobulk = hd.pseudobulk(mod="prot")
    assert pseudobulk.shape == (hd.n_groups, 3)
    with pytest.raises(ValueError, match="mod="):
        hd.pseudobulk()


def test_obsm_and_obsp_follow_selections(hd):
    embedding = pd.DataFrame(np.arange(hd.n_groups * 2).reshape(-1, 2), index=hd.obs_names, columns=["a", "b"])
    hd.obsm["emb"] = embedding
    hd.obsp["dist"] = np.arange(hd.n_groups**2, dtype=float).reshape(hd.n_groups, hd.n_groups)
    sub = hd.filter_groups(["P5", "P2"])
    assert sub.obs_names.tolist() == ["P2", "P5"]
    assert sub.obsm["emb"].index.tolist() == ["P2", "P5"]
    np.testing.assert_array_equal(sub.obsp["dist"], hd.obsp["dist"][np.ix_([2, 5], [2, 5])])
    assert hd.n_groups == 8


def test_filter_groups_selection_kinds(hd):
    by_query = hd.filter_groups("age > 40 and sex == 'F'")
    expected = hd.obs_names[(hd.obs["age"] > 40) & (hd.obs["sex"] == "F")]
    assert by_query.obs_names.tolist() == expected.tolist()
    by_callable = hd.filter_groups(lambda obs: obs["disease"] == "covid")
    assert set(by_callable.C.obs["patient"]) == set(hd.obs_names[hd.obs["disease"] == "covid"])
    by_mask = hd.filter_groups(np.arange(hd.n_groups) < 3)
    assert by_mask.n_groups == 3
    with pytest.raises(ValueError, match="length"):
        hd.filter_groups(np.array([True, False]))


def test_filter_inplace(hd):
    assert hd.filter_groups("site == 'A'", inplace=True) is None
    assert set(hd.obs["site"]) == {"A"}
    validate(hd)


def test_boolean_series_masks_align_by_index(cells, patients):
    hd = HiData.from_anndata(cells, group_id="patient")
    assert not hd.C.obs_names.equals(cells.obs_names)

    b_cells = hd.filter_cells(cells.obs["cell_type"] == "B")
    assert set(b_cells.C.obs_names) == set(cells.obs_names[cells.obs["cell_type"] == "B"])

    reversed_mask = (patients["age"] > 40).iloc[::-1]
    old = hd.filter_groups(reversed_mask)
    assert set(old.G.index) == set(patients.index[patients["age"] > 40])

    from_superset = b_cells.filter_cells(cells.obs["n_counts"] > 0)
    assert from_superset.C.obs_names.equals(b_cells.C.obs_names)

    with pytest.raises(ValueError, match="missing"):
        hd.filter_cells(cells.obs["cell_type"].iloc[:5] == "B")


def test_filter_cells_drops_empty_groups(hd):
    only_p0 = hd.filter_cells(lambda obs: obs["patient"] == "P0")
    assert only_p0.obs_names.tolist() == ["P0"]
    b_cells = hd.filter_cells("cell_type == 'B'")
    assert set(b_cells.C.obs["cell_type"]) == {"B"}
    validate(b_cells)


def test_table_G_has_no_variables(hd):
    with pytest.raises(IndexError, match="group table"):
        hd[:, ["x"]]
    assert hd.shape[1] == 0


def test_filter_groups_by_cells(hd):
    counts = hd._cell_group_counts("cell_type")
    assert "DC" not in counts.columns
    kept = hd.filter_groups_by_cells(5, groupby="cell_type", groups=["B", "T"])
    expected = counts.index[(counts[["B", "T"]] >= 5).all(axis=1)]
    assert kept.obs_names.tolist() == expected.tolist()
    total = hd.filter_groups_by_cells(20)
    assert (total.n_cells_per_group >= 20).all()
    with pytest.raises(KeyError):
        hd.filter_groups_by_cells(1, groupby="cell_type", groups=["DC"])


def test_subsample_cells(hd):
    sub = hd.subsample_cells(7)
    np.testing.assert_array_equal(sub.n_cells_per_group.to_numpy(), np.minimum(hd.n_cells_per_group.to_numpy(), 7))
    stratified = hd.subsample_cells(2, groupby="cell_type", random_state=3)
    assert (stratified._cell_group_counts("cell_type").to_numpy() <= 2).all()
    assert stratified.n_groups == hd.n_groups
    half = hd.subsample_cells(fraction=0.5)
    assert half.n_cells < hd.n_cells
    with pytest.raises(ValueError):
        hd.subsample_cells()


def test_composition(hd):
    fractions = hd.composition("cell_type")
    assert "cell_type_fractions" in hd.obsm
    np.testing.assert_allclose(fractions.sum(axis=1), 1)
    counts = hd.composition("cell_type", normalize=False)
    np.testing.assert_array_equal(counts.sum(axis=1).to_numpy(), hd.n_cells_per_group.to_numpy())
    assert hd.filter_groups(["P1"]).obsm["cell_type_counts"].shape == (1, counts.shape[1])


def test_push_and_pull_obs(hd, patients):
    hd.push_obs(["age", "sex"])
    expected = patients.loc[hd.C.obs["patient"], "age"].to_numpy()
    np.testing.assert_array_equal(hd.C.obs["age"].to_numpy(), expected)
    assert isinstance(hd.C.obs["sex"].dtype, pd.CategoricalDtype)
    with pytest.raises(ValueError, match="already exist"):
        hd.push_obs("age")

    hd.C.obs["batch"] = hd.C.obs["patient"].map({d: f"b{i % 2}" for i, d in enumerate(hd.obs_names)})
    assert "batch" in hd.pull_obs(drop=True)
    assert "batch" not in hd.C.obs.columns
    assert hd.obs.loc["P1", "batch"] == "b1"

    with pytest.raises(ValueError, match="varies within groups"):
        hd.pull_obs("n_counts")
    hd.pull_obs("n_counts", func="median")
    expected = hd.C.obs.groupby("patient", observed=True)["n_counts"].median()
    np.testing.assert_allclose(hd.obs["n_counts"], expected.loc[hd.obs_names])


def test_push_obsm_and_to_anndata(hd, cells):
    hd.obsm["emb"] = np.eye(hd.n_groups)[:, :3]
    hd.push_obsm("emb")
    np.testing.assert_array_equal(hd.C.obsm["emb"], hd.obsm["emb"][hd.group_codes])
    flat = hd.to_anndata(obsm=["emb"])
    assert {"age", "sex", "disease"} <= set(flat.obs.columns)
    assert flat.shape == hd.C.shape
    np.testing.assert_array_equal(flat.obsm["emb"], hd.C.obsm["emb"])
    assert "age" not in hd.C.obs.columns
    original = cells.obs.loc[flat.obs_names, "age"]
    np.testing.assert_array_equal(flat.obs["age"].to_numpy(), original.to_numpy())


def test_G_setter_keeps_group_matrices(hd):
    hd.obsp["dist"] = np.arange(hd.n_groups**2, dtype=float).reshape(hd.n_groups, hd.n_groups)
    old_dist, old_names = hd.obsp["dist"].copy(), hd.obs_names.copy()
    hd.G = hd.G[hd.G["age"] > hd.G["age"].median()].assign(flag=True)
    positions = old_names.get_indexer(hd.obs_names)
    np.testing.assert_array_equal(hd.obsp["dist"], old_dist[np.ix_(positions, positions)])
    assert set(hd.C.obs["patient"]) == set(hd.obs_names)
    assert hd.obs["flag"].all()
    validate(hd)


def test_C_setter_restricts_groups(hd):
    hd.C = hd.C[hd.C.obs["patient"].isin(["P3", "P4"]).to_numpy()]
    assert hd.obs_names.tolist() == ["P3", "P4"]
    with pytest.raises(TypeError):
        hd.C = hd.obs


def test_aggregate_matrices_and_obs(hd, cells):
    hd.aggregate(key_added="mean", sync_var=True)
    for group in ["P0", "P4"]:
        expected = np.asarray(cells[cells.obs["patient"] == group].X.mean(axis=0)).ravel()
        np.testing.assert_allclose(hd.obsm["mean"].loc[group].to_numpy(), expected, rtol=1e-6)
    hd.aggregate(layer="counts", func="sum", key_added="sum_T", filter_key="cell_type", filter_value="T")
    t_cells = cells[(cells.obs["patient"] == "P2") & (cells.obs["cell_type"] == "T")]
    np.testing.assert_allclose(hd.obsm["sum_T"].loc["P2"], np.asarray(t_cells.layers["counts"].sum(axis=0)).ravel())
    hd.aggregate(obsm="X_pca", func="var", key_added="pca_var")
    expected = cells.obsm["X_pca"][(cells.obs["patient"] == "P1").to_numpy()].var(axis=0, ddof=1)
    np.testing.assert_allclose(hd.obsm["pca_var"][1], expected)
    hd.aggregate(obs="n_counts", func="max", add_to_obs=True)
    assert hd.obs.loc["P3", "n_counts"] == cells.obs.loc[cells.obs["patient"] == "P3", "n_counts"].max()
    genes = hd[..., ["gene3", "gene1"]]
    assert genes.obsm["mean"].columns.tolist() == ["gene3", "gene1"]


@pytest.mark.parametrize("func", ["sum", "mean", "var", "count_nonzero", "median", "min", "max"])
def test_aggregation_functions_match_pandas(hd, func):
    from hidata._core._utils import aggregate_rows

    dense = hd.C.X.toarray()
    frame = pd.DataFrame(dense).groupby(hd.group_codes)
    expected = {
        "sum": frame.sum(),
        "mean": frame.mean(),
        "var": frame.var(),
        "count_nonzero": frame.agg(lambda x: (x != 0).sum()),
        "median": frame.median(),
        "min": frame.min(),
        "max": frame.max(),
    }[func]
    for X in (hd.C.X, dense):
        np.testing.assert_allclose(aggregate_rows(X, hd.group_codes, hd.n_groups, func), expected.to_numpy(), rtol=1e-5)


def test_pseudobulk(hd, cells):
    bulk = hd.pseudobulk(layer="counts")
    assert bulk.shape == (hd.n_groups, cells.n_vars)
    assert {"age", "sex", "patient", "n_cells"} <= set(bulk.obs.columns)
    np.testing.assert_allclose(bulk.X.sum(), cells.layers["counts"].sum())
    by_type = hd.pseudobulk(layer="counts", groupby="cell_type", min_cells=3)
    assert (by_type.obs["n_cells"] >= 3).all()
    row = by_type.obs.iloc[0]
    members = (cells.obs["patient"] == row["patient"]) & (cells.obs["cell_type"] == row["cell_type"])
    np.testing.assert_allclose(by_type.X[0], np.asarray(cells.layers["counts"][members.to_numpy()].sum(axis=0)).ravel())


def test_collapse(hd):
    hd.obsm["emb"] = pd.DataFrame(np.arange(hd.n_groups, dtype=float)[:, None], index=hd.obs_names, columns=["e"])
    sites = hd.collapse("site", func="mean")
    assert sites.obs_names.tolist() == ["A", "B"]
    assert sites.group_id == "site"
    assert sites.obs["n_patient"].sum() == hd.n_groups
    assert "bmi" in sites.obs.columns
    expected = hd.obsm["emb"]["e"].groupby(hd.obs["site"].to_numpy()).mean()
    np.testing.assert_allclose(sites.obsm["emb"]["e"].to_numpy(), expected.to_numpy())
    assert sites.n_cells == hd.n_cells
    assert "patient" in sites.C.obs.columns
    validate(sites)


def test_train_test_split_and_kfold(hd):
    train, test = hd.train_test_split(0.25, stratify="disease", random_state=1)
    assert not set(train.obs_names) & set(test.obs_names)
    assert train.n_groups + test.n_groups == hd.n_groups
    assert test.n_groups == 2
    assert set(test.obs["disease"]) == {"healthy", "covid"}
    assert set(train.C.obs["patient"]).isdisjoint(test.C.obs["patient"])
    seen = []
    for fold_train, fold_test in hd.kfold(4, stratify="disease"):
        assert not set(fold_train.obs_names) & set(fold_test.obs_names)
        seen.extend(fold_test.obs_names)
    assert sorted(seen) == sorted(hd.obs_names)
    with pytest.raises(ValueError):
        hd.train_test_split(0.0)


def test_copy_is_independent(hd):
    hd.uns["note"] = {"a": 1}
    copied = hd.copy()
    copied.obs["new"] = 1
    copied.uns["note"]["a"] = 2
    copied.C.obs["x"] = 0
    assert "new" not in hd.obs.columns
    assert hd.uns["note"]["a"] == 1
    assert "x" not in hd.C.obs.columns


def test_validate_detects_corruption(hd):
    validate(hd)
    broken = hd.copy()
    broken._C = broken._C[::-1].copy()
    with pytest.raises(HiDataSchemaError, match="grouped"):
        validate(broken)


def test_repr(hd, hd_genotypes):
    text = repr(hd)
    assert "DataFrame with 8 groups" in text
    assert "columns:" in text
    assert "AnnData object" in repr(hd_genotypes)
    assert str(hd).startswith("HiData(n_G_obs=8")
    assert "\x1b[" not in text
    html = hd._repr_html_()
    assert html.startswith("<pre") and "HiData(n_groups=8" in html


def test_anndata_G_obs_operations(hd_genotypes):
    hd = hd_genotypes
    assert "age" in hd.obs.columns
    sub = hd.filter_groups("age > 30")
    assert sub.G.n_obs == sub.n_groups
    hd.push_obs("age")
    assert "age" in hd.C.obs.columns
    hd.composition("cell_type")
    assert "cell_type_fractions" in hd.G.obsm


def test_empty_selection_raises(hd):
    with pytest.raises(ValueError, match="no groups"):
        hd.filter_groups("age > 1000")


def test_pull_obs_with_renaming(hd):
    names = hd.pull_obs({"n_counts": "median_n_counts"}, func="median")
    assert names == ["median_n_counts"]
    assert "n_counts" in hd.C.obs.columns
    expected = hd.C.obs.groupby("patient", observed=True)["n_counts"].median()
    np.testing.assert_allclose(hd.obs["median_n_counts"], expected.loc[hd.obs_names])


def test_pseudobulk_becomes_group_side(hd):
    hd.obsm["emb"] = np.arange(hd.n_groups * 2.0).reshape(-1, 2)
    hd.obsp["dist"] = np.eye(hd.n_groups)
    hd.uns["subgroup_representations"] = ["emb"]
    bulk = hd.pseudobulk(func="mean")
    np.testing.assert_array_equal(bulk.obsm["emb"], hd.obsm["emb"])
    assert bulk.uns["subgroup_representations"] == ["emb"]
    bulk_obs = bulk.obs.copy()
    hd.G = bulk
    assert hd.G_type == "anndata"
    np.testing.assert_array_equal(hd.G.X, bulk.X)
    hd.obsp["other"] = np.zeros((hd.n_groups, hd.n_groups))
    assert "other" not in bulk.obsp
    pd.testing.assert_frame_equal(bulk.obs, bulk_obs)
    subset = hd.filter_groups("age > 40")
    assert subset.G.shape == (subset.n_groups, hd.C.n_vars)
    validate(subset)


def test_group_level_is_only_called_G(cells, genotypes):
    hd = HiData(G=genotypes, C=cells, group_id="patient")
    assert not hasattr(hd, "D") and not hasattr(hd, "D_type")
    with pytest.raises(TypeError):
        HiData(D=genotypes, C=cells, group_id="patient")


def test_selecting_groups_keeps_the_categories_of_group_level_anndata():
    rng = np.random.default_rng(0)
    var = pd.DataFrame(
        {"a0": pd.Categorical(["A", "C", "G", "T"] * 5), "chrom": pd.Categorical(["1"] * 10 + ["2"] * 10)},
        index=[f"v{j}" for j in range(20)],
    )
    G = ad.AnnData(
        X=rng.integers(0, 3, size=(6, 20)).astype(np.int8),
        obs=pd.DataFrame({"sex": pd.Categorical(["F", "M"] * 3)}, index=[f"d{i}" for i in range(6)]),
        var=var,
    )
    C = ad.AnnData(
        X=np.zeros((12, 1)),
        obs=pd.DataFrame({"group_id": [f"d{i % 6}" for i in range(12)]}, index=[f"c{i}" for i in range(12)]),
    )
    hd = HiData(G=G, C=C, group_id="group_id")
    sub = hd.sel(G_obs=[0, 2])
    assert list(sub.G.var["a0"].cat.categories) == ["A", "C", "G", "T"]
    assert list(sub.G.obs["sex"].cat.categories) == ["F", "M"]
    assert list(hd.sel(G_var=[0, 1]).G.var["chrom"].cat.categories) == ["1", "2"]
    np.testing.assert_array_equal(sub.G.X, G.X[[0, 2]])


def _genotype_hd(names=None):
    names = names or [f"d{i}" for i in range(6)]
    var = pd.DataFrame(
        {"a0": pd.Categorical(["A", "C"] * 10, categories=["A", "C", "G", "T"])}, index=[f"v{j}" for j in range(20)]
    )
    G = ad.AnnData(
        X=np.zeros((6, 20), dtype=np.int8),
        obs=pd.DataFrame({"sex": pd.Categorical(["F"] * 6, categories=["F", "M"])}, index=names),
        var=var,
    )
    G.obsm["pcs"] = np.arange(12, dtype=float).reshape(6, 2)
    obs = pd.DataFrame(
        {"group_id": [f"d{i % 5}" for i in range(12)], "cell_type": ["T", "B"] * 6}, index=[f"c{i}" for i in range(12)]
    )
    C = ad.AnnData(X=np.ones((12, 3), dtype=np.float32), obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))
    return G, C


GROUP_LEVEL_ROUTES = {
    "cells only": lambda hd: hd[..., hd.C.obs["cell_type"] == "T", :],
    "genes only": lambda hd: hd.sel(C_var=["g0", "g1"]),
    "filter_cells": lambda hd: hd.filter_cells(hd.C.obs["cell_type"] == "T"),
    "sel of groups": lambda hd: hd.sel(G_obs=["d0", "d2"]),
    "filter_groups": lambda hd: hd.filter_groups(["d1"]),
    "pseudobulk": lambda hd: hd.pseudobulk(),
}


@pytest.mark.parametrize("route", GROUP_LEVEL_ROUTES.values(), ids=GROUP_LEVEL_ROUTES.keys())
def test_group_level_anndata_keeps_its_categories_on_every_route(route):
    G, C = _genotype_hd()
    out = route(HiData(G=G, C=C, group_id="group_id"))
    if isinstance(out, HiData):
        assert list(out.G.var["a0"].cat.categories) == ["A", "C", "G", "T"]
        assert list(out.G.obs["sex"].cat.categories) == ["F", "M"]


@pytest.mark.parametrize("route", [None, *GROUP_LEVEL_ROUTES.values()], ids=["build", *GROUP_LEVEL_ROUTES.keys()])
@pytest.mark.parametrize(
    "names", [None, ["d0", "d1", "d2", "d3", "d4", "nan"]], ids=["named", "a group without a name"]
)
def test_views_of_group_level_anndata_never_prune_categories(monkeypatch, route, names):
    G, C = _genotype_hd(names)
    pruning = []
    init_as_view = ad.AnnData._init_as_view

    def record(self, adata_ref, *args, **kwargs):
        if adata_ref.n_vars == G.n_vars:
            pruning.append(ad.settings.remove_unused_categories)
        return init_as_view(self, adata_ref, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "_init_as_view", record)
    hd = HiData(G=G, C=C, group_id="group_id")
    if route is not None:
        route(hd)
    assert not any(pruning)


def _reordered_hd():
    """HiData whose subgroups and cells the constructor reorders, so position and label disagree.

    Group ``a`` has subgroups s1 and s3, group ``b`` has s0 and s2; cells c1 and c4 belong to s0.
    """
    groups = pd.DataFrame(index=pd.Index(["a", "b"], name="group_id"))
    subgroups = pd.DataFrame(
        {"group_id": ["b", "a", "b", "a"]}, index=pd.Index(["s0", "s1", "s2", "s3"], name="subgroup_id")
    )
    obs = pd.DataFrame({"subgroup_id": ["s2", "s0", "s3", "s1", "s0", "s3"]}, index=[f"c{i}" for i in range(6)])
    cells = AnnData(np.zeros((6, 1)), obs=obs)
    return HiData(G=groups, S=subgroups, C=cells, group_id="group_id", subgroup_id="subgroup_id"), cells, subgroups


def test_reordered_hd_reorders_subgroups_and_cells():
    hd, cells, subgroups = _reordered_hd()
    assert not hd.C.obs_names.equals(cells.obs_names)
    assert not hd.levels["subgroup"].obs_names.equals(subgroups.index)


CELL_SELECTIONS = {
    "filter_cells": lambda hd, key: hd.filter_cells(key),
    "sel": lambda hd, key: hd.sel(C_obs=key),
    "getitem": lambda hd, key: hd[:, :, key],
}
CELL_KEYS = {
    "Series in input order": lambda hd, cells: pd.Series(cells.obs_names.isin(["c1", "c4"]), index=cells.obs_names),
    "Series in reversed order": lambda hd, cells: pd.Series(cells.obs_names.isin(["c1", "c4"]), index=cells.obs_names)[
        ::-1
    ],
    "numpy mask in hd order": lambda hd, cells: hd.C.obs_names.isin(["c1", "c4"]),
    "labels": lambda hd, cells: ["c1", "c4"],
}


@pytest.mark.parametrize("select", CELL_SELECTIONS.values(), ids=CELL_SELECTIONS.keys())
@pytest.mark.parametrize("make_key", CELL_KEYS.values(), ids=CELL_KEYS.keys())
def test_cell_selection_keeps_the_named_cells(select, make_key):
    hd, cells, _ = _reordered_hd()
    result = select(hd, make_key(hd, cells))
    assert sorted(result.C.obs_names) == ["c1", "c4"]
    assert list(result.levels["subgroup"].obs_names) == ["s0"]
    assert list(result.obs_names) == ["b"]


GROUP_SELECTIONS = {
    "filter_groups": lambda hd, key: hd.filter_groups(key),
    "sel": lambda hd, key: hd.sel(G_obs=key),
    "getitem": lambda hd, key: hd[key],
}


@pytest.mark.parametrize("select", GROUP_SELECTIONS.values(), ids=GROUP_SELECTIONS.keys())
def test_group_selection_aligns_a_boolean_series_by_label(select):
    hd, _, _ = _reordered_hd()
    keep = pd.Series([True, False], index=["b", "a"])  # reversed relative to hd.obs_names
    result = select(hd, keep)
    assert list(result.obs_names) == ["b"]
    assert sorted(result.levels["subgroup"].obs_names) == ["s0", "s2"]


SUBGROUP_SELECTIONS = {
    "filter_subgroups": lambda hd, key: hd.filter_subgroups(key),
    "sel": lambda hd, key: hd.sel(S_obs=key),
}


@pytest.mark.parametrize("select", SUBGROUP_SELECTIONS.values(), ids=SUBGROUP_SELECTIONS.keys())
def test_subgroup_selection_aligns_a_boolean_series_by_label(select):
    hd, _, subgroups = _reordered_hd()
    keep = pd.Series(subgroups.index == "s0", index=subgroups.index)  # in input order, not hd order
    result = select(hd, keep)
    assert list(result.levels["subgroup"].obs_names) == ["s0"]
    assert sorted(result.C.obs_names) == ["c1", "c4"]


def test_boolean_series_with_unknown_labels_raises():
    hd, cells, _ = _reordered_hd()
    keep = pd.Series(True, index=[f"x{i}" for i in range(cells.n_obs)])
    with pytest.raises(ValueError, match="missing 6 of 6 labels"):
        hd.sel(C_obs=keep)


def test_boolean_series_without_names_raises_and_warns_about_position():
    hd, cells, _ = _reordered_hd()
    keep = pd.Series(cells.obs_names.isin(["c1", "c4"]))  # index 0..5, not cell names
    for select in CELL_SELECTIONS.values():
        with pytest.raises(ValueError, match=r"default index 0, 1, 2.*\.values.*reorders"):
            select(hd, keep)


def test_variable_selection_aligns_a_boolean_series_by_label():
    cells = AnnData(
        np.zeros((2, 3)),
        obs=pd.DataFrame({"group_id": ["a", "b"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    groups = AnnData(np.zeros((2, 3)), obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["v0", "v1", "v2"]))
    hd = HiData(G=groups, C=cells, group_id="group_id")
    reversed_genes = pd.Series([True, False, False], index=["g2", "g1", "g0"])
    reversed_variants = pd.Series([True, False, False], index=["v2", "v1", "v0"])
    result = hd.sel(G_var=reversed_variants, C_var=reversed_genes)
    assert list(result.G.var_names) == ["v2"]
    assert list(result.C.var_names) == ["g2"]
    with pytest.raises(ValueError, match="cell variables has the default index"):
        hd.sel(C_var=pd.Series([True, False, False]))


def _numbered_hd():
    """Like ``_reordered_hd``, with a different value in every cell and subgroup so that mixed-up rows show."""
    hd, cells, subgroups = _reordered_hd()
    cells = AnnData(np.arange(6.0).reshape(6, 1), obs=cells.obs.assign(number=np.arange(6)))
    subgroups = subgroups.assign(number=np.arange(4))
    groups = pd.DataFrame({"number": [0, 1]}, index=pd.Index(["a", "b"], name="group_id"))
    return HiData(G=groups, S=subgroups, C=cells, group_id="group_id", subgroup_id="subgroup_id"), cells


def _state(hd):
    """Everything a selection can change: the names and values at every level and how the levels are linked."""
    subgroups = hd.levels["subgroup"]
    return {
        "groups": list(hd.obs_names),
        "group numbers": hd.obs["number"].tolist(),
        "subgroups": list(subgroups.obs_names),
        "subgroup numbers": subgroups.obs["number"].tolist(),
        "subgroup groups": subgroups.obs["group_id"].astype(str).tolist(),
        "cells": list(hd.C.obs_names),
        "cell numbers": hd.C.obs["number"].tolist(),
        "cell values": np.asarray(hd.C.X).ravel().tolist(),
        "cell subgroups": hd.C.obs["subgroup_id"].astype(str).tolist(),
    }


def _inplace(method):
    def select(hd, key):
        getattr(hd, method)(key, inplace=True)
        return hd

    return select


# Every way to select each level. All of them must give the same object for the same selection.
ROUTES = {
    "group": {
        "filter_groups": lambda hd, key: hd.filter_groups(key),
        "filter_groups inplace": _inplace("filter_groups"),
        "sel": lambda hd, key: hd.sel(G_obs=key),
        "getitem": lambda hd, key: hd[key],
        "getitem with ellipsis": lambda hd, key: hd[key, ...],
    },
    "subgroup": {
        "filter_subgroups": lambda hd, key: hd.filter_subgroups(key),
        "filter_subgroups inplace": _inplace("filter_subgroups"),
        "sel": lambda hd, key: hd.sel(S_obs=key),
    },
    "cell": {
        "filter_cells": lambda hd, key: hd.filter_cells(key),
        "filter_cells inplace": _inplace("filter_cells"),
        "sel": lambda hd, key: hd.sel(C_obs=key),
        "getitem": lambda hd, key: hd[:, :, key],
        "getitem with ellipsis": lambda hd, key: hd[..., key, :],
    },
}
LEVEL_NAMES = {
    "group": lambda hd: hd.obs_names,
    "subgroup": lambda hd: hd.levels["subgroup"].obs_names,
    "cell": lambda hd: hd.C.obs_names,
}
KEPT = {"group": ["b"], "subgroup": ["s0", "s3"], "cell": ["c1", "c4", "c5"]}
# Every form of the same selection, built from the names of the level in the order of the HiData.
KEY_FORMS = {
    "Series in hd order": lambda names, kept: pd.Series(names.isin(kept), index=names),
    "Series in reversed order": lambda names, kept: pd.Series(names.isin(kept), index=names)[::-1],
    "Series in sorted order": lambda names, kept: pd.Series(names.isin(kept), index=names).sort_index(),
    "Series with extra names": lambda names, kept: pd.Series(
        [*names.isin(kept), True], index=[*names, "not in the object"]
    ),
    "nullable boolean Series": lambda names, kept: pd.Series(names.isin(kept), index=names, dtype="boolean")[::-1],
    "numpy mask": lambda names, kept: names.isin(kept),
    "list of booleans": lambda names, kept: names.isin(kept).tolist(),
    "list of names": lambda names, kept: list(kept),
    "numpy array of names": lambda names, kept: np.array(kept),
    "Index of names": lambda names, kept: pd.Index(kept),
}


@pytest.mark.parametrize("level", ROUTES)
@pytest.mark.parametrize("form", KEY_FORMS)
def test_every_selection_route_gives_the_same_result(level, form):
    reference, cells = _numbered_hd()
    names, kept = LEVEL_NAMES[level](reference), KEPT[level]
    expected = _state(reference.sel(**{f"{level[0].upper()}_obs": names.isin(kept)}))
    assert expected[f"{level}s"] == [name for name in names if name in kept]
    assert expected["cell values"] == cells[expected["cells"]].X.ravel().tolist()  # values follow the names

    for route, select in ROUTES[level].items():
        hd, _ = _numbered_hd()
        key = KEY_FORMS[form](LEVEL_NAMES[level](hd), kept)
        assert _state(select(hd, key)) == expected, f"{route} with {form}"


@pytest.mark.parametrize("level", ROUTES)
def test_names_in_another_order_select_the_same_units(level):
    """``filter_*`` keeps the order of the object. ``sel`` and ``hd[...]`` follow anndata and take the order of the
    names, within the grouping by group and subgroup. Both must keep the same units with the same values."""
    reference, cells = _numbered_hd()
    kept = KEPT[level][::-1]
    for route, select in ROUTES[level].items():
        hd, _ = _numbered_hd()
        state = _state(select(hd, kept))
        assert sorted(state[f"{level}s"]) == sorted(kept), route
        assert state["cell values"] == cells[state["cells"]].X.ravel().tolist(), route
        assert state["cell subgroups"] == cells[state["cells"]].obs["subgroup_id"].tolist(), route
        expected_groups = reference.levels["subgroup"].obs["group_id"].astype(str)[state["subgroups"]].tolist()
        assert state["subgroup groups"] == expected_groups, route


@pytest.mark.parametrize("level", ROUTES)
def test_query_and_callable_give_the_same_result_as_a_mask(level):
    reference, _ = _numbered_hd()
    names, kept = LEVEL_NAMES[level](reference), KEPT[level]
    expected = _state(reference.sel(**{f"{level[0].upper()}_obs": names.isin(kept)}))
    filter_level = f"filter_{level}s"
    numbers = {"group": [1], "subgroup": [0, 3], "cell": [1, 4, 5]}[level]
    for key in (f"number in {numbers}", lambda table: table["number"].isin(numbers)):
        hd, _ = _numbered_hd()
        assert _state(getattr(hd, filter_level)(key)) == expected


@pytest.mark.parametrize("level", ROUTES)
def test_every_selection_route_rejects_a_series_without_names(level):
    for select in ROUTES[level].values():
        hd, _ = _numbered_hd()
        key = pd.Series(LEVEL_NAMES[level](hd).isin(KEPT[level]))  # index 0, 1, 2, ...
        with pytest.raises(ValueError, match="default index 0, 1, 2"):
            select(hd, key)


def test_unnamed_series_raises_even_if_the_cells_are_called_0_1_2():
    """Cells without real names are called "0", "1", ...: the labels of an unnamed Series would match them."""
    obs = pd.DataFrame({"group_id": ["b", "a", "b", "a"], "n_counts": [10, 20, 30, 40]})
    hd = HiData.from_anndata(AnnData(np.zeros((4, 1)), obs=obs), "group_id", columns=[])
    assert list(hd.C.obs_names) == ["1", "3", "0", "2"]  # reordered
    in_object_order = hd.C.obs["n_counts"].to_numpy() >= 30
    for select in CELL_SELECTIONS.values():
        with pytest.raises(ValueError, match="default index 0, 1, 2"):
            select(hd, pd.Series(in_object_order))
        assert sorted(select(hd, in_object_order).C.obs["n_counts"]) == [30, 40]  # an array goes by position
        assert sorted(select(hd, hd.C.obs["n_counts"] >= 30).C.obs["n_counts"]) == [30, 40]  # named "0", "1", ...


@pytest.mark.parametrize("select", GROUP_SELECTIONS.values(), ids=GROUP_SELECTIONS.keys())
def test_missing_entries_of_a_mask_count_as_false(select):
    groups = pd.DataFrame(
        {"age": pd.array([61, pd.NA, 40], dtype="Int64"), "smoker": pd.array([True, pd.NA, False], dtype="boolean")},
        index=pd.Index(["a", "b", "c"], name="group_id"),
    )
    obs = pd.DataFrame({"group_id": ["a", "b", "c"]}, index=["c0", "c1", "c2"])
    hd = HiData(G=groups, C=AnnData(np.zeros((3, 1)), obs=obs), group_id="group_id")
    older = hd.obs["age"] > 50
    assert older.isna().tolist() == [False, True, False]
    assert list(select(hd, older).obs_names) == ["a"]
    assert list(select(hd, hd.obs["age"] <= 50).obs_names) == ["c"]
    assert list(select(hd, hd.obs["smoker"]).obs_names) == ["a"]


FILTERS = {"group": "filter_groups", "subgroup": "filter_subgroups", "cell": "filter_cells"}


@pytest.mark.parametrize("level", ROUTES)
def test_every_selection_route_rejects_unknown_names(level):
    for select in ROUTES[level].values():
        hd, _ = _numbered_hd()
        with pytest.raises(KeyError, match="not in the object"):
            select(hd, [KEPT[level][0], "not in the object"])


@pytest.mark.parametrize("level", FILTERS)
@pytest.mark.parametrize("key", [[0, 1], np.array([0, 1]), 0, np.int64(1), ["s0", 1]], ids=repr)
def test_filters_reject_integers(level, key):
    hd, _ = _numbered_hd()
    with pytest.raises(TypeError, match=r"integers.*pass them as strings.*hd\[\.\.\.\]"):
        getattr(hd, FILTERS[level])(key)


@pytest.mark.parametrize("level", FILTERS)
def test_filters_reject_slices(level):
    hd, _ = _numbered_hd()
    with pytest.raises(TypeError, match=r"slice.*hd\[\.\.\.\]"):
        getattr(hd, FILTERS[level])(slice(0, 2))


def test_numeric_identifiers_are_selected_as_text():
    groups = pd.DataFrame(index=pd.Index([101, 102], name="group_id"))
    obs = pd.DataFrame({"group_id": [101, 102]}, index=["c0", "c1"])
    hd = HiData(G=groups, C=AnnData(np.zeros((2, 1)), obs=obs), group_id="group_id")
    assert list(hd.filter_groups(["102"]).obs_names) == ["102"]
    with pytest.raises(TypeError, match="integers"):
        hd.filter_groups([102])


def _levels_hd(layout):
    """Six cells of groups a, b, c, with G as a table or an AnnData and an optional subgroup table."""
    obs = pd.DataFrame(
        {
            "group_id": np.repeat(["a", "b", "c"], 2),
            "subgroup_id": ["s1", "s1", "s2", "s3", "s4", "s4"],
            "cell_type": ["T", "B"] * 3,
        },
        index=[f"cell{i}" for i in range(6)],
    )
    C = AnnData(np.arange(18, dtype=np.float32).reshape(6, 3), obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))
    groups = pd.DataFrame({"age": [30.0, 40.0, 50.0]}, index=pd.Index(["a", "b", "c"], name="group_id"))
    G = AnnData(np.ones((3, 2), dtype=np.float32), obs=groups) if layout.startswith("anndata") else groups
    if layout.endswith("subgroups"):
        subgroups = pd.DataFrame(
            {"group_id": ["a", "b", "b", "c"]}, index=pd.Index(["s1", "s2", "s3", "s4"], name="subgroup_id")
        )
        return HiData(G=G, S=subgroups, C=C, group_id="group_id", subgroup_id="subgroup_id")
    return HiData(G=G, C=C, group_id="group_id")


SELECTIONS = {
    "filter_cells keeping every group": lambda hd: hd.filter_cells(hd.C.obs["cell_type"] == "T"),
    "filter_groups keeping every cell": lambda hd: hd.filter_groups(["a", "b", "c"]),
    "sel of genes": lambda hd: hd.sel(C_var=["g0", "g1"]),
    "hd[...] of genes": lambda hd: hd[..., ["g0", "g1"]],
    "subsample_cells": lambda hd: hd.subsample_cells(5),
    "train_test_split": lambda hd: hd.train_test_split(test_size=1, random_state=0)[0],
}


@pytest.mark.parametrize("select", SELECTIONS.values(), ids=SELECTIONS.keys())
@pytest.mark.parametrize("layout", ["table", "anndata", "table with subgroups", "anndata with subgroups"])
def test_writing_to_a_selection_leaves_the_original_unchanged(select, layout):
    hd = _levels_hd(layout)
    hd.uns["nested"] = {"key": 1}
    before = hd.copy()
    new = select(hd)
    (new.G.obs if isinstance(new.G, AnnData) else new.G)["added"] = 1
    new.composition("cell_type")
    new.C.obs["added"] = 1
    new.C.X[0, 0] = -1
    new.C.X = new.C.X * 0
    if isinstance(new.G, AnnData):
        new.G.X = new.G.X * 0
    new.uns["nested"]["key"] = 2
    for level in new.level_names:
        new.levels[level].obsm["added"] = np.zeros((new.levels[level].n_obs, 1))
    for level in hd.level_names:
        assert "added" not in hd.levels[level].obsm
        assert "added" not in hd.levels[level].obs.columns
    assert list(hd.obsm.keys()) == list(before.obsm.keys())
    np.testing.assert_array_equal(hd.C.X, before.C.X)
    if isinstance(hd.G, AnnData):
        np.testing.assert_array_equal(hd.G.X, before.G.X)
    assert hd.uns["nested"] == {"key": 1}
    validate(hd)


def test_slicing_genes_keeps_synced_matrices_of_the_original():
    hd = _levels_hd("table")
    hd._var_dims_to_sync = ["X_mean"]
    hd.aggregate(key_added="X_mean", sync_var=True)
    columns = list(hd.obsm["X_mean"].columns)
    sliced = hd[..., ["g0", "g1"]]
    assert list(sliced.obsm["X_mean"].columns) == ["g0", "g1"]
    assert list(hd.obsm["X_mean"].columns) == columns == ["g0", "g1", "g2"]


REJECTED = {
    "C with groups contradicting the subgroups": ("C", lambda hd: hd.C.obs.assign(group_id="c")),
    "S with duplicate subgroup names": ("S", None),
    "G without the groups of the cells": ("G", None),
}


@pytest.mark.parametrize("what", REJECTED.keys())
def test_a_rejected_level_assignment_keeps_the_object(what):
    hd = _levels_hd("table with subgroups")
    before = hd.copy()
    if what.startswith("C"):
        bad = hd.C.copy()
        bad.obs["group_id"] = "c"
        with pytest.raises(ValueError, match="disagrees"):
            hd.C = bad
    elif what.startswith("S"):
        bad = pd.DataFrame({"group_id": ["a", "b"]}, index=pd.Index(["s1", "s1"], name="subgroup_id"))
        with pytest.raises(ValueError):
            hd.S = bad
    else:
        with pytest.raises(ValueError):
            hd.G = pd.DataFrame({"age": [1.0]}, index=pd.Index(["zz"], name="group_id"))
    for level in hd.level_names:
        assert hd.levels[level].obs.equals(before.levels[level].obs)
    np.testing.assert_array_equal(hd.C.X, before.C.X)
    validate(hd)


def test_assigning_X_of_a_level_never_writes_into_another_object():
    obs = pd.DataFrame({"group_id": ["b", "a", "b", "a"]}, index=[f"c{i}" for i in range(4)])
    adata = AnnData(np.ones((4, 2), dtype=np.float32), obs=obs)
    hd = HiData.from_anndata(adata, group_id="group_id")
    hd.C.X = np.full((4, 2), 7, dtype=np.float32)
    np.testing.assert_array_equal(adata.X, 1)
    subset = hd.filter_groups(["a"])
    subset.C.X = np.zeros(subset.C.shape, dtype=np.float32)
    np.testing.assert_array_equal(hd.C.X, 7)
    validate(hd)


def _two_group_cells(groups=("A", "A", "B", "B"), X=None):
    obs = pd.DataFrame({"group": list(groups)}, index=[f"c{i}" for i in range(len(groups))])
    X = np.ones((len(groups), 2), dtype=np.float32) if X is None else X
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1"]))


def test_mudata_modalities_are_matched_to_the_cells_by_name():
    obs = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=["c0", "c1", "c2", "c3"])
    rna = AnnData(X=np.array([[1.0], [2.0], [10.0], [20.0]]), obs=obs.copy())
    mdata = md.MuData({"rna": rna})
    mdata.obs["group"] = obs["group"]
    groups = pd.DataFrame(index=pd.Index(["B", "A"], name="group"))
    hd = HiData(G=groups, C=mdata, group_id="group")
    hd.aggregate(mod="rna", func="sum", key_added="total")
    assert hd.obsm["total"].iloc[:, 0].to_dict() == {"B": 30.0, "A": 3.0}
    assert hd.get_df("rna:0", level="cell")["rna:0"].to_dict() == {"c2": 10.0, "c3": 20.0, "c0": 1.0, "c1": 2.0}
    hd.filter_groups(["B"]).aggregate(mod="rna", func="sum", key_added="total")


def test_synced_pseudobulk_of_a_modality_keeps_its_own_variables():
    obs = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=list("wxyz"))
    rna = AnnData(X=np.arange(8.0).reshape(4, 2), obs=obs.copy(), var=pd.DataFrame(index=["g0", "g1"]))
    prot = AnnData(X=np.ones((4, 1)), obs=pd.DataFrame(index=obs.index), var=pd.DataFrame(index=["CD3"]))
    mdata = md.MuData({"rna": rna, "prot": prot})
    mdata.obs["group"] = obs["group"]
    hd = HiData(G=pd.DataFrame(index=pd.Index(["A", "B"], name="group")), C=mdata, group_id="group")
    hd.aggregate(mod="rna", key_added="rna_mean", sync_var=True)
    assert list(hd.filter_groups(["A"]).obsm["rna_mean"].columns) == ["g0", "g1"]


@pytest.mark.parametrize("extension", [".hd.h5", ".hd.zarr"])
@pytest.mark.parametrize("ids", [[1, 2], [1.0, 2.0], pd.Categorical(["1", "2"])], ids=["int", "float", "categorical"])
def test_identifier_columns_of_any_type_are_stored_as_text(tmp_path, extension, ids):
    groups = pd.DataFrame({"group": ids, "age": [30, 40]})
    cells = _two_group_cells(groups=[1, 1, 2, 2])
    hd = HiData(G=groups, C=cells, group_id="group")
    assert list(hd.obs_names) == ["1", "2"] and list(hd.obs["group"]) == ["1", "2"]
    hd.write_hd(tmp_path / f"ids{extension}")
    assert list(read_hd(tmp_path / f"ids{extension}").obs_names) == ["1", "2"]


def test_whole_numbers_stored_as_floats_match_integer_identifiers():
    groups = pd.DataFrame(index=pd.Index([1, 2], name="group"))
    cells = _two_group_cells(groups=[1.0, 1.0, 2.0, np.nan])
    hd = HiData(G=groups, C=cells, group_id="group")
    assert list(hd.obs_names) == ["1", "2"] and hd.n_cells == 3


def test_get_df_gives_one_column_per_time_point_for_an_ehrdata_variable():
    ed = pytest.importorskip("ehrdata", exc_type=ImportError)
    values = np.arange(12.0).reshape(2, 2, 3)
    groups = ed.EHRData(
        X=values,
        obs=pd.DataFrame(index=pd.Index(["A", "B"], name="group")),
        var=pd.DataFrame(index=["crp", "il6"]),
        tem=pd.DataFrame(index=["t0", "t1", "t2"]),
    )
    hd = HiData(G=groups, C=_two_group_cells(), group_id="group")
    frame = hd.get_df("crp", level="group")
    assert list(frame.columns) == ["crp_t0", "crp_t1", "crp_t2"]
    np.testing.assert_array_equal(frame.to_numpy(), values[:, 0, :])


def test_nest_flatten_and_collapse_do_not_share_cells_with_the_original():
    groups = pd.DataFrame({"patient": ["P1", "P2"]}, index=pd.Index(["A", "B"], name="group"))
    hd = HiData(G=groups, C=_two_group_cells(), group_id="group")
    nested = hd.nest("patient")
    nested.C.obs["qc"] = "x"
    nested.C.X[:] = 0
    assert "qc" not in hd.C.obs.columns and hd.C.X.sum() == 8
    flat = nested.flatten()
    flat.C.uns["note"] = 1
    assert "note" not in nested.C.uns
    collapsed = hd.collapse("patient")
    collapsed.C.X[:] = 0
    assert hd.C.X.sum() == 8


def test_the_objects_passed_to_the_constructor_are_not_changed():
    groups = AnnData(X=np.ones((2, 3)), obs=pd.DataFrame({"group": ["A", "B"]}, index=["row0", "row1"]))
    HiData(G=groups, C=_two_group_cells(), group_id="group", obsm={"pcs": np.zeros((2, 2))})
    assert list(groups.obs_names) == ["row0", "row1"] and list(groups.obsm) == []


def test_from_anndata_takes_a_missing_group_from_the_subgroup():
    obs = pd.DataFrame(
        {"subgroup": ["s1", "s1", "s2", "s2"], "group": ["A", np.nan, "B", "B"]}, index=[f"c{i}" for i in range(4)]
    )
    hd = HiData.from_anndata(AnnData(np.ones((4, 1)), obs=obs), group_id="group", subgroup_id="subgroup")
    assert list(hd.obs_names) == ["A", "B"] and hd.n_cells == 4
    assert list(hd.S["group"]) == ["A", "B"]
    obs.loc["c1", "group"] = "B"
    with pytest.raises(ValueError, match="several groups"):
        HiData.from_anndata(AnnData(np.ones((4, 1)), obs=obs), group_id="group", subgroup_id="subgroup")


@pytest.mark.parametrize("subgroup_id", [None, "subgroup"])
def test_from_anndata_orders_groups_by_their_categories(subgroup_id):
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(["P10", "P2", "P1", "P1"], categories=["P1", "P2", "P10"]),
            "subgroup": ["s10", "s2", "s1", "s1"],
        },
        index=[f"c{i}" for i in range(4)],
    )
    if subgroup_id is not None:
        obs.loc["c3", "group"] = np.nan
    hd = HiData.from_anndata(AnnData(np.ones((4, 1)), obs=obs), group_id="group", subgroup_id=subgroup_id)
    assert list(hd.obs_names) == ["P1", "P2", "P10"] and hd.n_cells == 4


def test_names_can_be_given_as_a_set():
    hd = HiData(G=pd.DataFrame(index=pd.Index(["A", "B"], name="group")), C=_two_group_cells(), group_id="group")
    assert list(hd.filter_groups({"A"}).obs_names) == ["A"]


def test_replacing_the_group_table_keeps_the_matrices_of_the_groups_kept():
    groups = pd.DataFrame({"age": [30, 40, 50]}, index=pd.Index(["A", "B", "C"], name="group"))
    hd = HiData(G=groups, C=_two_group_cells(), group_id="group")
    hd.aggregate(func="mean", key_added="pb")
    hd.G = groups.assign(bmi=[22.0, 25.0, 30.0])
    assert list(hd.obsm) == ["pb"] and list(hd.obs_names) == ["A", "B"]
    validate(hd)
