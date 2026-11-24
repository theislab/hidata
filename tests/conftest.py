import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

N_GROUPS = 8
N_GENES = 12
N_SNPS = 5


@pytest.fixture
def patients():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 80, N_GROUPS),
            "sex": pd.Categorical(np.tile(["F", "M"], N_GROUPS // 2)),
            "disease": pd.Categorical(np.repeat(["healthy", "covid"], N_GROUPS // 2)),
            "site": pd.Categorical(np.tile(["A", "A", "B", "B"], N_GROUPS // 4)),
            "bmi": rng.normal(25, 3, N_GROUPS),
        },
        index=pd.Index([f"P{i}" for i in range(N_GROUPS)], name="patient"),
    )


@pytest.fixture
def cells(patients):
    rng = np.random.default_rng(0)
    sizes = rng.integers(10, 30, N_GROUPS)
    groups = np.repeat(patients.index.to_numpy(), sizes)
    groups = groups[rng.permutation(groups.size)]
    n_cells = groups.size
    obs = pd.DataFrame(
        {
            "patient": groups,
            "cell_type": pd.Categorical(rng.choice(["T", "B", "NK"], n_cells), categories=["B", "NK", "T", "DC"]),
            "n_counts": rng.integers(500, 5000, n_cells),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    obs = obs.join(patients, on="patient")
    counts = sp.random(
        n_cells, N_GENES, density=0.4, format="csr", random_state=0, data_rvs=lambda n: rng.integers(1, 20, n)
    )
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"gene{i}" for i in range(N_GENES)]),
        layers={"counts": counts.copy()},
        obsm={"X_pca": rng.normal(size=(n_cells, 4))},
    )
    return adata


@pytest.fixture
def genotypes(patients):
    rng = np.random.default_rng(2)
    return ad.AnnData(
        X=rng.binomial(2, 0.3, size=(N_GROUPS, N_SNPS)).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(patients.index, name="patient")),
        var=pd.DataFrame(
            {"chrom": "1", "pos": np.arange(1, N_SNPS + 1), "a0": "A", "a1": "G"},
            index=[f"snp{i}" for i in range(N_SNPS)],
        ),
    )


@pytest.fixture
def hd(cells):
    from hidata import HiData

    return HiData.from_anndata(cells, group_id="patient")


@pytest.fixture
def hd_genotypes(cells, genotypes, patients):
    from hidata import HiData

    cells = cells.copy()
    cells.obs = cells.obs.drop(columns=patients.columns)
    genotypes.obs = genotypes.obs.join(patients)
    return HiData(G=genotypes, C=cells, group_id="patient")


@pytest.fixture
def visits(cells, patients):
    rng = np.random.default_rng(3)
    cells = cells.copy()
    n_visits = {p: 1 + (i % 3) for i, p in enumerate(patients.index)}
    visit = np.array([rng.integers(0, n_visits[p]) for p in cells.obs["patient"]])
    cells.obs["subgroup"] = [f"{p}_v{v}" for p, v in zip(cells.obs["patient"], visit, strict=True)]
    cells.obs["day"] = visit * 7
    cells.obs["tissue"] = np.where(visit % 2 == 0, "blood", "marrow")
    return cells


@pytest.fixture
def hd3(visits):
    from hidata import HiData

    return HiData.from_anndata(visits, group_id="patient", subgroup_id="subgroup")


@pytest.fixture
def ehrdata():
    try:
        import ehrdata
    except (ImportError, AttributeError) as e:
        pytest.skip(f"ehrdata cannot be imported: {e}")
    return ehrdata
