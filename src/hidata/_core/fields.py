from dataclasses import dataclass


@dataclass(frozen=True)
class GAnn:
    """Default names of group-level fields."""

    group: str = "group_id"
    group_family: str = "group_family_id"
    group_paternal_id: str = "group_paternal_id"
    group_maternal_id: str = "group_maternal_id"
    group_sex: str = "sex"
    GROUP_X_KEY: str = "group_X"
    GROUP_LABELS_KEY: str = "group_labels"
    GROUP_BATCH_KEY: str = "group_batch"
    GROUP_CAT_COVS_KEY: str = "group_extra_categorical_covs"
    GROUP_CONT_COVS_KEY: str = "group_extra_continuous_covs"
    GROUP_INDICES_KEY: str = "group_ind_x"
    GROUP_ID_KEY: str = "group_id"


@dataclass(frozen=True)
class CAnn:
    """Default names of cell-level fields."""

    celltype: str = "celltype"
    CELL_X_KEY: str = "cell_X"
    CELL_LABELS_KEY: str = "cell_labels"
    CELL_BATCH_KEY: str = "cell_batch"
    CELL_CAT_COVS_KEY: str = "cell_extra_categorical_covs"
    CELL_CONT_COVS_KEY: str = "cell_extra_continuous_covs"
    CELL_INDICES_KEY: str = "cell_ind_x"
    CELL_GROUP_KEY: str = "group_id"
