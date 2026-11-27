import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from hidata import HiData


def _constant_within_group_hd() -> HiData:
    groups = pd.DataFrame({"age": [30, 40]}, index=pd.Index(["P0", "P1"], name="patient"))
    cells = ad.AnnData(
        X=np.array([[2.0, 1.0], [2.0, 3.0], [5.0, 1.0], [5.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "patient": ["P0", "P0", "P1", "P1"],
                "site": ["A", "A", "B", "B"],
                "const_count": [2, 2, 5, 5],
                "qc_pass": [True, True, False, False],
            },
            index=["c0", "c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["const_gene", "var_gene"]),
    )
    return HiData(G=groups, C=cells, group_id="patient")


def test_get_df_applies_func_to_constant_numeric_and_bool_columns():
    hd = _constant_within_group_hd()

    summed = hd.get_df(["const_gene", "var_gene", "const_count", "qc_pass", "site"], level="group", func="sum")
    expected_bulk = hd.pseudobulk(func="sum")

    np.testing.assert_allclose(summed["const_gene"].to_numpy(), expected_bulk[:, "const_gene"].X.ravel())
    np.testing.assert_allclose(summed["var_gene"].to_numpy(), expected_bulk[:, "var_gene"].X.ravel())
    np.testing.assert_array_equal(summed["const_count"].to_numpy(), [4, 10])
    np.testing.assert_array_equal(summed["qc_pass"].to_numpy(), [2, 0])
    assert summed["site"].tolist() == ["A", "B"]

    variances = hd.get_df("const_gene", level="group", func="var")
    np.testing.assert_allclose(variances["const_gene"].to_numpy(), [0.0, 0.0])


def test_pull_obs_applies_func_to_constant_numeric_and_bool_columns():
    hd = _constant_within_group_hd()

    hd.pull_obs({"const_count": "total_count", "qc_pass": "n_passed", "site": "site"}, func="sum")
    np.testing.assert_array_equal(hd.obs["total_count"].to_numpy(), [4, 10])
    np.testing.assert_array_equal(hd.obs["n_passed"].to_numpy(), [2, 0])
    assert hd.obs["site"].tolist() == ["A", "B"]


def test_get_df_partial_mudata_constant_gene_sum():
    groups = pd.DataFrame({"age": [30, 40]}, index=pd.Index(["P0", "P1"], name="patient"))
    rna = ad.AnnData(
        X=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        obs=pd.DataFrame({"patient": ["P0", "P0", "P1", "P1"]}, index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g0"]),
    )
    atac = ad.AnnData(
        X=np.array([[5], [5]], dtype=np.int64),
        obs=pd.DataFrame({"patient": ["P0", "P0"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["peak0"]),
    )
    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.obs["patient"] = ["P0", "P0", "P1", "P1"]
    hd = HiData(G=groups, C=mdata, group_id="patient")

    out = hd.get_df("atac:peak0", level="group", func="sum")
    assert out.loc["P0", "atac:peak0"] == 10.0
    assert np.isnan(out.loc["P1", "atac:peak0"])
