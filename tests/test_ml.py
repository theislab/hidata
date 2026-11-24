import numpy as np
import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

import torch.nn.functional as F
from torch.utils.data import DataLoader

from hidata.ml import GroupMILModel, MILDataset, mil_collate_fn


def test_variable_size_bags(hd):
    dataset = MILDataset(hd, group_labels_key="disease", cell_batch_key="cell_type")
    assert len(dataset) == hd.n_groups
    assert dataset.encodings["disease"] == ["covid", "healthy"]
    item = dataset[3]
    group = hd.obs_names[3]
    assert item["group_id"] == group
    np.testing.assert_allclose(item["cell_x"].numpy(), hd.cells_of(group).X.toarray())
    assert item["group_x"] is None
    assert item["cell_mask"] is None
    assert item["group_y"].item() == dataset.encodings["disease"].index(hd.obs.loc[group, "disease"])
    batch = mil_collate_fn([dataset[0], dataset[1]])
    assert batch["group_y"].shape == (2,)
    assert isinstance(batch["cell_x"], list)
    assert batch["group_id"] == list(hd.obs_names[:2])


def test_fixed_size_bags_and_loader(hd):
    hd.obsm["emb"] = np.random.default_rng(0).normal(size=(hd.n_groups, 3))
    dataset = MILDataset(
        hd,
        group_layer="emb",
        group_labels_key="age",
        cell_layer="X_pca",
        cell_labels_key="cell_type",
        n_cells=15,
        random_state=0,
        split_groups=["P1", "P6", "P2"],
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=mil_collate_fn)
    batch = next(iter(loader))
    assert batch["cell_x"].shape == (3, 15, 4)
    assert batch["cell_mask"].shape == (3, 15)
    assert batch["group_x"].shape == (3, 3)
    sizes = hd.n_cells_per_group.loc[["P1", "P6", "P2"]].to_numpy()
    np.testing.assert_array_equal(batch["cell_mask"].sum(1).numpy(), np.minimum(sizes, 15))
    np.testing.assert_allclose(batch["group_y"].numpy(), hd.obs.loc[["P1", "P6", "P2"], "age"].to_numpy())
    padded = ~batch["cell_mask"]
    assert torch.all(batch["cell_x"][padded] == 0)


def test_subsampled_bag_comes_from_group(hd):
    dataset = MILDataset(hd, cell_layer="X_pca", n_cells=5, random_state=1)
    item = dataset[0]
    own = hd.cells_of(hd.obs_names[0]).obsm["X_pca"]
    for row in item["cell_x"].numpy():
        assert np.isclose(own, row).all(axis=1).any()


def test_missing_explicit_key_raises(hd):
    with pytest.raises(KeyError, match="nope"):
        MILDataset(hd, group_labels_key="nope")
    with pytest.raises(KeyError):
        MILDataset(hd, split_groups=["P0", "unknown"])
    with pytest.raises(ValueError):
        MILDataset(hd, split_groups=["P0"], split_indices=[0])


def test_genotype_group_features(hd_genotypes):
    dataset = MILDataset(hd_genotypes, split_indices=[0, 2])
    item = dataset[1]
    np.testing.assert_allclose(item["group_x"].numpy(), hd_genotypes.G.X[2])


def test_subgroup_level_bags_and_balanced_weights(hd3):
    dataset = MILDataset(hd3, level="subgroup", group_labels_key="disease", cell_layer="X_pca")
    assert len(dataset) == hd3.n_subgroups
    item = dataset[0]
    assert item["cell_x"].shape[0] == hd3.n_cells_per_subgroup.iloc[0]
    weights = dataset.balanced_weights()
    labels = dataset._group_fields["group_y"]
    totals = {label: weights[labels == label].sum().item() for label in np.unique(labels)}
    np.testing.assert_allclose(list(totals.values()), 1.0)


def test_group_features_of_a_mudata_group_level_follow_the_groups():
    import mudata as md
    import pandas as pd
    from anndata import AnnData

    from hidata import HiData

    first = AnnData(X=np.array([[0.0], [1.0]]), obs=pd.DataFrame(index=["D0", "D1"]))
    second = AnnData(X=np.array([[10.0], [0.0]]), obs=pd.DataFrame(index=["D1", "D0"]))
    groups = md.MuData({"a": first, "b": second})
    groups.obs.index.name = "group"
    cells = AnnData(X=np.ones((4, 2)), obs=pd.DataFrame({"group": ["D0", "D0", "D1", "D1"]}, index=list("wxyz")))
    dataset = MILDataset(HiData(G=groups, C=cells, group_id="group"))
    features = {dataset[i]["group_id"]: dataset[i]["group_x"].tolist() for i in range(2)}
    assert features == {"D0": [0.0, 0.0], "D1": [1.0, 10.0]}


def _trainer():
    return pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


@pytest.mark.parametrize("n_cells", [None, 8])
def test_model_trains_on_bags_of_a_hidata(hd_genotypes, n_cells):
    dataset = MILDataset(
        hd_genotypes, group_labels_key="disease", cell_batch_key="cell_type", n_cells=n_cells, split_indices=[0, 1, 2]
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=mil_collate_fn)
    model = GroupMILModel(n_input_group=hd_genotypes.G.n_vars, n_input_cell=hd_genotypes.C.n_vars)
    _trainer().fit(model, loader)


def _batch(n_groups=2, n_cells=4, n_group_x=3, n_cell_x=5, y=None):
    gen = torch.Generator().manual_seed(0)
    return {
        "group_x": torch.randn(n_groups, n_group_x, generator=gen),
        "cell_x": [torch.randn(n_cells, n_cell_x, generator=gen) for _ in range(n_groups)],
        "group_y": torch.zeros(n_groups, 1) if y is None else y,
    }


def test_forward_ignores_padded_cells():
    torch.manual_seed(0)
    model = GroupMILModel(n_input_group=3, n_input_cell=5).eval()
    batch = _batch(n_groups=1, n_cells=3)
    padded = dict(batch)
    padded["cell_x"] = [torch.cat([batch["cell_x"][0], torch.zeros(2, 5)])]
    padded["cell_mask"] = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model(padded))


def test_classification_uses_cross_entropy():
    torch.manual_seed(0)
    model = GroupMILModel(n_input_group=3, n_input_cell=5, n_output=3).eval()
    batch = _batch(y=torch.tensor([[0.0], [2.0]]))  # (B, 1), as MILDataset + mil_collate_fn stack it
    with torch.no_grad():
        expected = F.cross_entropy(model(batch), torch.tensor([0, 2]))
        torch.testing.assert_close(model._loss(batch), expected)


def test_regression_loss_does_not_broadcast():
    torch.manual_seed(0)
    model = GroupMILModel(n_input_group=3, n_input_cell=5).eval()
    y = torch.tensor([[1.0], [-1.0]])
    batch = _batch(y=y)
    with torch.no_grad():
        expected = F.mse_loss(model(batch).squeeze(-1), y.squeeze(-1))
        torch.testing.assert_close(model._loss(batch), expected)


@pytest.mark.parametrize("n_output", [1, 3])
def test_training_runs_for_regression_and_classification(n_output):
    gen = torch.Generator().manual_seed(0)
    items = [
        {
            "group_id": f"P{i}",
            "group_x": torch.randn(3, generator=gen),
            "group_y": torch.tensor([float(i % max(n_output, 2))]),
            "cell_x": torch.randn(4 + i, 5, generator=gen),
        }
        for i in range(6)
    ]
    loader = DataLoader(items, batch_size=2, collate_fn=mil_collate_fn)
    model = GroupMILModel(n_input_group=3, n_input_cell=5, n_output=n_output)
    trainer = _trainer()
    trainer.fit(model, loader, loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
