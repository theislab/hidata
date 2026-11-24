import numpy as np
import pytest

import hidata
from hidata import concat, validate


def test_concat_tables_with_label(hd):
    hd.obsm["emb"] = np.arange(hd.n_groups * 2.0).reshape(-1, 2)
    first, second = hd.filter_groups(["P0", "P1"]), hd.filter_groups(["P5"])
    merged = concat({"cohort_a": first, "cohort_b": second}, label="cohort")
    assert merged.obs_names.tolist() == ["P0", "P1", "P5"]
    assert merged.obs["cohort"].tolist() == ["cohort_a", "cohort_a", "cohort_b"]
    assert merged.n_cells == first.n_cells + second.n_cells
    np.testing.assert_array_equal(merged.obsm["emb"], hd.obsm["emb"][[0, 1, 5]])
    validate(merged)


def test_concat_anndata_G(hd_genotypes):
    merged = concat([hd_genotypes.filter_groups(["P2"]), hd_genotypes.filter_groups(["P3", "P4"])])
    assert merged.G_type == "anndata"
    assert merged.shape[:2] == (3, hd_genotypes.shape[1])


def test_concat_rejects_duplicates_and_mixed_kinds(hd, hd_genotypes):
    with pytest.raises(ValueError, match="not unique"):
        concat([hd, hd])
    with pytest.raises(ValueError, match="kinds"):
        concat([hd.filter_groups(["P0"]), hd_genotypes.filter_groups(["P1"])])


def test_cell_level_obs_filter(hd):
    out = hidata.pp.cell_level_obs_filter(hd, "cell_type", ["B", "NK"], inplace=False)
    assert set(out.C.obs["cell_type"]) <= {"B", "NK"}
    assert not out.C.is_view
    n_before = hd.n_cells
    assert hidata.pp.cell_level_obs_filter(hd, "cell_type", "T") is None
    assert set(hd.C.obs["cell_type"]) == {"T"}
    assert hd.n_cells < n_before


def test_group_level_filters(hd, hd_genotypes):
    out = hidata.pp.group_level_obs_filter(hd, "disease", "covid", inplace=False, copy=False)
    assert set(out.obs["disease"]) == {"covid"}
    hidata.pp.group_level_obs_filter(hd, "site", np.array(["B"]))
    assert set(hd.obs["site"]) == {"B"}
    hidata.pp.group_level_var_filter(hd_genotypes, group_level_key="pos", group_level_values=[1, 2])
    assert hd_genotypes.shape[1] == 2
    with pytest.raises(ValueError, match="no variables"):
        hidata.pp.group_level_var_filter(hd, group_level_key="pos", group_level_values=1)


def _cohort(prefix, genes):
    import pandas as pd
    from anndata import AnnData

    from hidata import HiData

    cells = AnnData(
        X=np.ones((2, len(genes))),
        obs=pd.DataFrame({"group": [prefix + "1"] * 2}, index=[prefix + "c0", prefix + "c1"]),
        var=pd.DataFrame(index=genes),
    )
    hd = HiData(G=pd.DataFrame(index=pd.Index([prefix + "1"], name="group")), C=cells, group_id="group")
    hd.aggregate(func="mean", key_added="pb")
    return hd


def test_concat_outer_keeps_every_column_of_group_matrices():
    merged = concat([_cohort("a", ["g0", "g1", "g2"]), _cohort("b", ["g0", "g1"])], join="outer")
    assert list(merged.obsm["pb"].columns) == ["g0", "g1", "g2"]
    assert np.isnan(merged.obsm["pb"].loc["b1", "g2"])


def test_concat_rejects_ehrdata_levels():
    ed = pytest.importorskip("ehrdata", exc_type=ImportError)
    import pandas as pd
    from anndata import AnnData

    from hidata import HiData

    def cohort(prefix):
        groups = ed.EHRData(
            X=np.ones((1, 1, 2)),
            obs=pd.DataFrame(index=pd.Index([prefix + "1"], name="group")),
            tem=pd.DataFrame(index=["t0", "t1"]),
        )
        cells = AnnData(np.ones((1, 1)), obs=pd.DataFrame({"group": [prefix + "1"]}, index=[prefix + "c"]))
        return HiData(G=groups, C=cells, group_id="group")

    with pytest.raises(ValueError, match="EHRData"):
        concat([cohort("a"), cohort("b")])
