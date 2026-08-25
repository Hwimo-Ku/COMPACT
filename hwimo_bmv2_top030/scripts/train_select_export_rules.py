#!/usr/bin/env python3
"""
Train COMPACT Proposed models and an MTL-all Low-K baseline, then export
BMv2 ternary table commands.

Pipeline
1. Train one single-task tree per task.
2. Compute STL mean-absolute SHAP and used split-feature sets.
3. Enumerate all task partitions and select one Proposed cluster set by:
      ResourcePool -> Field-AbsSHAP compatibility
4. Proposed:
      - select Top-K bit features per selected cluster
      - train one multi-output tree per cluster
      - save existing-pipeline artifacts
      - export one combined BMv2 command file
5. MTL-all Low-K:
      - select the globally lowest-K SHAP bits
      - train one seven-output tree
      - save existing-pipeline artifacts
      - export one BMv2 command file

Default outputs for K=30
  models/top_030/cluster_*/
    model_mtl_feature_selected_top_30.pkl
    results.json
    selected_feature_indices.npy
    selected_feature_indices.json
    selected_feature_indices.csv
    rules.json

  models/top_030/selected_cluster_set.json
  commands/Hwimo_top030_proposed_commands.txt

  models/low_030/cluster_1_2_3_4_5_6_7/
    model_mtl_feature_selected_low_30.pkl
    results.json
    selected_feature_indices.npy
    selected_feature_indices.json
    selected_feature_indices.csv
    rules.json

  models/low_030/mtl_all_low_summary.json
  commands/Hwimo_low030_mtl_all_commands.txt
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

import numpy as np

if "bool" not in np.__dict__:
    setattr(np, "bool", np.bool_)

if "int" not in np.__dict__:
    setattr(np, "int", int)

warnings.filterwarnings(
    "ignore",
    message="In the future `np.bool`",
)

try:
    import shap
except ModuleNotFoundError as exc:
    raise SystemExit(
        "[ERROR] Python package 'shap' is not installed.\n"
        "Install it with: python3 -m pip install --user shap"
    ) from exc

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


TASK_NAME_BY_NUM = {
    1: "vpn",
    2: "tor",
    3: "service",
    4: "app",
    5: "flow_size",
    6: "flow_duration",
    7: "avg_pkt_len",
}

FEATURE_LENGTHS_169 = {
    "ip_len": 16,
    "ip_proto": 8,
    "ip_src": 32,
    "ip_dst": 32,
    "tcp_sport": 16,
    "tcp_dport": 16,
    "tcp_flags": 6,
    "ip_id": 16,
    "ip_ttl": 8,
    "ip_flags": 3,
    "tcp_window": 16,
}

FEATURE_START_169 = {
    "ip_len": 0,
    "ip_proto": 16,
    "ip_src": 24,
    "ip_dst": 56,
    "tcp_sport": 88,
    "tcp_dport": 104,
    "tcp_flags": 120,
    "ip_id": 126,
    "ip_ttl": 142,
    "ip_flags": 150,
    "tcp_window": 153,
}

FIELD_ORDER = [
    "ip_len",
    "ip_proto",
    "ip_src",
    "ip_dst",
    "tcp_sport",
    "tcp_dport",
    "tcp_flags",
    "ip_ttl",
    "ip_flags",
    "tcp_window",
]

GLOBAL_WIDTH = 169
COMPACT_WIDTH = 153

REMOVED_GLOBAL_START = 126
REMOVED_GLOBAL_END_EXCL = 142

COMPACT_TO_GLOBAL = (
    list(range(0, REMOVED_GLOBAL_START))
    + list(
        range(
            REMOVED_GLOBAL_END_EXCL,
            GLOBAL_WIDTH,
        )
    )
)

BIT_TO_FIELD_169: Dict[int, str] = {}

for field_name, start in FEATURE_START_169.items():
    for bit_idx in range(
        start,
        start + FEATURE_LENGTHS_169[field_name],
    ):
        BIT_TO_FIELD_169[bit_idx] = field_name

BIT_TO_FIELD: Dict[int, str] = {
    compact_idx: BIT_TO_FIELD_169[global_idx]
    for compact_idx, global_idx
    in enumerate(COMPACT_TO_GLOBAL)
}

MASKS_169 = {
    1:
        "0" * 16
        + "0" * 8
        + "1" * 32
        + "1" * 32
        + "0" * 16
        + "1" * 16
        + "0" * 6
        + "0" * 16
        + "1" * 8
        + "1" * 3
        + "1" * 16,

    2:
        "1" * 16
        + "0" * 8
        + "1" * 32
        + "1" * 32
        + "1" * 16
        + "1" * 16
        + "0" * 6
        + "0" * 16
        + "0" * 8
        + "0" * 3
        + "1" * 16,

    3:
        "0" * 16
        + "0" * 8
        + "1" * 32
        + "0" * 32
        + "1" * 16
        + "1" * 16
        + "0" * 6
        + "0" * 16
        + "1" * 8
        + "1" * 3
        + "1" * 16,

    4:
        "0" * 16
        + "0" * 8
        + "1" * 32
        + "0" * 32
        + "1" * 16
        + "1" * 16
        + "0" * 6
        + "0" * 16
        + "1" * 8
        + "1" * 3
        + "1" * 16,

    5:
        "1" * 16
        + "1" * 8
        + "0" * 32
        + "0" * 32
        + "0" * 16
        + "1" * 16
        + "1" * 6
        + "0" * 16
        + "0" * 8
        + "1" * 3
        + "1" * 16,

    6:
        "1" * 16
        + "1" * 8
        + "0" * 32
        + "1" * 32
        + "1" * 16
        + "0" * 16
        + "1" * 6
        + "0" * 16
        + "0" * 8
        + "0" * 3
        + "1" * 16,

    7:
        "1" * 16
        + "0" * 8
        + "1" * 32
        + "1" * 32
        + "0" * 16
        + "0" * 16
        + "0" * 6
        + "0" * 16
        + "1" * 8
        + "1" * 3
        + "1" * 16,
}


def compact_mask_from_169(
    mask_169: str,
) -> str:
    if len(mask_169) != GLOBAL_WIDTH:
        raise ValueError(
            f"expected 169-bit mask, "
            f"got {len(mask_169)}"
        )

    return "".join(
        mask_169[global_idx]
        for global_idx in COMPACT_TO_GLOBAL
    )


MASKS = {
    task_id: compact_mask_from_169(mask)
    for task_id, mask in MASKS_169.items()
}


def default_root() -> Path:
    here = Path(__file__).resolve()

    if here.parent.name == "scripts":
        return here.parent.parent

    return here.parent


def canonical_cluster(
    cluster: Iterable[int],
) -> Tuple[int, ...]:
    out = tuple(
        sorted(int(x) for x in cluster)
    )

    for task_id in out:
        if task_id not in TASK_NAME_BY_NUM:
            raise ValueError(
                f"unknown task id: {task_id}"
            )

    return out


def canonical_cluster_set(
    cluster_set: Iterable[Iterable[int]],
) -> Tuple[Tuple[int, ...], ...]:
    clusters = [
        canonical_cluster(cluster)
        for cluster in cluster_set
    ]

    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                len(cluster),
                cluster,
            ),
        )
    )


def cluster_to_str(
    cluster: Iterable[int],
) -> str:
    return (
        "{"
        + ",".join(
            str(x)
            for x in canonical_cluster(cluster)
        )
        + "}"
    )


def cluster_set_to_str(
    cluster_set: Iterable[Iterable[int]],
) -> str:
    return " | ".join(
        cluster_to_str(cluster)
        for cluster
        in canonical_cluster_set(cluster_set)
    )


def cluster_slug(
    cluster: Sequence[int],
) -> str:
    return (
        "cluster_"
        + "_".join(
            str(task)
            for task in canonical_cluster(cluster)
        )
    )


def parse_task_set(
    value: str,
) -> List[int]:
    tasks = [
        int(item.strip())
        for item
        in value.replace("{", "").replace("}", "").split(",")
        if item.strip()
    ]

    if not tasks:
        raise ValueError(
            f"invalid task set: {value}"
        )

    if len(tasks) != len(set(tasks)):
        raise ValueError(
            f"duplicated task in task set: {value}"
        )

    for task in tasks:
        if task not in TASK_NAME_BY_NUM:
            raise ValueError(
                f"unknown task id: {task}"
            )

    return sorted(tasks)


def mask_to_indices(
    mask_str: str,
) -> List[int]:
    return [
        index
        for index, bit in enumerate(mask_str)
        if bit == "1"
    ]


def read_bitstrings(
    path: Path,
    width: int = COMPACT_WIDTH,
) -> List[str]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        rows = [
            line.strip().split()[0]
            for line in file
            if line.strip()
        ]

    widths = sorted(
        set(len(row) for row in rows)
    )

    if widths != [width]:
        raise ValueError(
            f"{path}: expected {width}-bit rows, "
            f"observed widths={widths}"
        )

    return rows


def read_label_file(
    path: Path,
) -> np.ndarray:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        values = [
            int(line.strip().split()[0])
            for line in file
            if line.strip()
        ]

    return np.asarray(
        values,
        dtype=np.int64,
    )


def bitstrings_to_numpy(
    bitstrings: Sequence[str],
    selected_indices: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [
            [
                1 if row[int(index)] == "1" else 0
                for index in selected_indices
            ]
            for row in bitstrings
        ],
        dtype=np.uint8,
    )


def set_partitions(
    items: Sequence[int],
):
    items = tuple(items)

    if not items:
        yield tuple()
        return

    first = items[0]

    for smaller in set_partitions(
        items[1:]
    ):
        for index in range(len(smaller)):
            new_block = tuple(
                sorted(
                    (first,)
                    + smaller[index]
                )
            )

            yield canonical_cluster_set(
                list(smaller[:index])
                + [new_block]
                + list(smaller[index + 1:])
            )

        yield canonical_cluster_set(
            ((first,),) + smaller
        )


def minmax(
    values: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    )

    low = float(np.min(array))
    high = float(np.max(array))

    if abs(high - low) <= 1e-12:
        return np.zeros_like(array)

    return (
        array - low
    ) / (
        high - low
    )


def cosine(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    norm_first = float(
        np.linalg.norm(first)
    )

    norm_second = float(
        np.linalg.norm(second)
    )

    if (
        norm_first <= 1e-12
        or norm_second <= 1e-12
    ):
        return 0.0

    return float(
        np.dot(first, second)
        / (
            norm_first
            * norm_second
        )
    )


def normalize_l1(
    vector: np.ndarray,
) -> np.ndarray:
    total = float(
        np.sum(
            np.abs(vector)
        )
    )

    if total <= 1e-12:
        return np.zeros_like(
            vector,
            dtype=float,
        )

    return (
        np.asarray(
            vector,
            dtype=float,
        )
        / total
    )


def make_tree(
    seed: int,
    depth: int | None,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
    )


def get_used_split_features(
    model: DecisionTreeClassifier,
    compact_feature_indices: Sequence[int],
) -> set[int]:
    used_local = np.unique(
        model.tree_.feature[
            model.tree_.feature >= 0
        ]
    ).astype(int)

    compact_feature_array = np.asarray(
        compact_feature_indices,
        dtype=int,
    )

    return set(
        int(value)
        for value
        in compact_feature_array[
            used_local
        ].tolist()
    )


def convert_shap_values_to_sample_matrix(
    shap_values: Any,
    pred_label: np.ndarray,
) -> np.ndarray:
    if isinstance(shap_values, list):
        n_samples = shap_values[0].shape[0]
        n_features = shap_values[0].shape[1]

        output = np.zeros(
            (
                n_samples,
                n_features,
            ),
            dtype=float,
        )

        for sample_index in range(n_samples):
            class_index = max(
                0,
                min(
                    int(pred_label[sample_index]),
                    len(shap_values) - 1,
                ),
            )

            output[sample_index] = (
                shap_values[class_index][sample_index]
            )

        return output

    array = np.asarray(shap_values)

    if array.ndim == 2:
        return array.astype(float)

    if array.ndim == 3:
        n_samples = array.shape[0]

        if (
            array.shape[1] > 1
            and array.shape[2] <= 100
        ):
            output = np.zeros(
                (
                    n_samples,
                    array.shape[1],
                ),
                dtype=float,
            )

            for sample_index in range(n_samples):
                class_index = max(
                    0,
                    min(
                        int(pred_label[sample_index]),
                        array.shape[2] - 1,
                    ),
                )

                output[sample_index] = (
                    array[
                        sample_index,
                        :,
                        class_index,
                    ]
                )

            return output

        if array.shape[1] <= 100:
            output = np.zeros(
                (
                    n_samples,
                    array.shape[2],
                ),
                dtype=float,
            )

            for sample_index in range(n_samples):
                class_index = max(
                    0,
                    min(
                        int(pred_label[sample_index]),
                        array.shape[1] - 1,
                    ),
                )

                output[sample_index] = (
                    array[
                        sample_index,
                        class_index,
                        :,
                    ]
                )

            return output

    raise ValueError(
        f"unsupported SHAP shape: {array.shape}"
    )


def compute_mean_abs_shap(
    model: DecisionTreeClassifier,
    X_eval: np.ndarray,
) -> np.ndarray:
    prediction = model.predict(X_eval)

    shap_values = (
        shap.TreeExplainer(model)
        .shap_values(X_eval)
    )

    sample_matrix = (
        convert_shap_values_to_sample_matrix(
            shap_values,
            prediction,
        )
    )

    return np.mean(
        np.abs(sample_matrix),
        axis=0,
    )


def compact_shap_to_field_vector(
    shap_by_compact_index: Dict[int, float],
    compact_feature_indices: Sequence[int],
) -> np.ndarray:
    by_field = {
        field: 0.0
        for field in FIELD_ORDER
    }

    for compact_index in compact_feature_indices:
        field = BIT_TO_FIELD[
            int(compact_index)
        ]

        by_field[field] += float(
            shap_by_compact_index.get(
                int(compact_index),
                0.0,
            )
        )

    return normalize_l1(
        np.asarray(
            [
                by_field[field]
                for field in FIELD_ORDER
            ],
            dtype=float,
        )
    )


def train_single_task_trees(
    active_tasks: Sequence[int],
    X_train: np.ndarray,
    X_test: np.ndarray,
    labels: Dict[int, Dict[str, np.ndarray]],
    compact_same_mask_indices: Sequence[int],
    seed: int,
    depth: int,
):
    stl_shap: Dict[
        int,
        Dict[int, float],
    ] = {}

    stl_used_features: Dict[
        int,
        set[int],
    ] = {}

    stl_field_vectors: Dict[
        int,
        np.ndarray,
    ] = {}

    for task_id in tqdm(
        active_tasks,
        desc="STL models",
        unit="task",
        dynamic_ncols=True,
    ):
        model = make_tree(
            seed=seed,
            depth=depth,
        )

        model.fit(
            X_train,
            labels[task_id]["train"],
        )

        mean_abs_local = compute_mean_abs_shap(
            model=model,
            X_eval=X_test,
        )

        shap_by_compact = {
            int(
                compact_same_mask_indices[
                    local_index
                ]
            ): float(score)
            for local_index, score
            in enumerate(mean_abs_local)
        }

        stl_shap[task_id] = shap_by_compact

        stl_used_features[task_id] = (
            get_used_split_features(
                model=model,
                compact_feature_indices=(
                    compact_same_mask_indices
                ),
            )
        )

        stl_field_vectors[task_id] = (
            compact_shap_to_field_vector(
                shap_by_compact_index=(
                    shap_by_compact
                ),
                compact_feature_indices=(
                    compact_same_mask_indices
                ),
            )
        )

    return (
        stl_shap,
        stl_used_features,
        stl_field_vectors,
    )


def cluster_resource_gain(
    cluster: Sequence[int],
    used: Dict[int, set[int]],
) -> float:
    cluster = canonical_cluster(cluster)

    separate_count = sum(
        len(used[task])
        for task in cluster
    )

    union_count = len(
        set().union(
            *(
                used[task]
                for task in cluster
            )
        )
    )

    return float(
        separate_count
        - union_count
    )


def cluster_compat_gain(
    cluster: Sequence[int],
    vectors: Dict[int, np.ndarray],
    mean_weight: float,
    singleton_policy: str,
) -> float:
    cluster = canonical_cluster(cluster)

    if len(cluster) == 1:
        return {
            "include_one": 1.0,
            "include_zero": 0.0,
            "exclude": math.nan,
        }[singleton_policy]

    similarities = [
        max(
            0.0,
            min(
                1.0,
                cosine(
                    vectors[first],
                    vectors[second],
                ),
            ),
        )
        for first, second
        in combinations(cluster, 2)
    ]

    return float(
        mean_weight
        * np.mean(similarities)
        + (
            1.0 - mean_weight
        )
        * np.min(similarities)
    )


def score_cluster_set(
    cluster_set: Sequence[Sequence[int]],
    used: Dict[int, set[int]],
    vectors: Dict[int, np.ndarray],
    mean_weight: float,
    singleton_policy: str,
) -> Dict[str, Any]:
    cluster_set = canonical_cluster_set(
        cluster_set
    )

    resource_gain = sum(
        cluster_resource_gain(
            cluster=cluster,
            used=used,
        )
        for cluster in cluster_set
    )

    weighted_sum = 0.0
    total_weight = 0.0

    for cluster in cluster_set:
        compatibility = cluster_compat_gain(
            cluster=cluster,
            vectors=vectors,
            mean_weight=mean_weight,
            singleton_policy=singleton_policy,
        )

        if math.isnan(compatibility):
            continue

        weight = float(
            len(cluster)
        )

        weighted_sum += (
            weight
            * compatibility
        )

        total_weight += weight

    if total_weight <= 1e-12:
        compatibility_gain = 0.0
    else:
        compatibility_gain = (
            weighted_sum
            / total_weight
        )

    return {
        "cluster_set": cluster_set,
        "cluster_set_str": (
            cluster_set_to_str(cluster_set)
        ),
        "resource_gain_saved_features": float(
            resource_gain
        ),
        "field_abs_shap_compat_gain": float(
            compatibility_gain
        ),
    }


def select_cluster_set(
    active_tasks: Sequence[int],
    used: Dict[int, set[int]],
    vectors: Dict[int, np.ndarray],
    lambda_value: float,
    mean_weight: float,
    singleton_policy: str,
) -> Dict[str, Any]:
    candidates = sorted(
        set(
            set_partitions(
                active_tasks
            )
        ),
        key=cluster_set_to_str,
    )

    rows = [
        score_cluster_set(
            cluster_set=cluster_set,
            used=used,
            vectors=vectors,
            mean_weight=mean_weight,
            singleton_policy=singleton_policy,
        )
        for cluster_set in candidates
    ]

    resource_normalized = minmax(
        [
            row[
                "resource_gain_saved_features"
            ]
            for row in rows
        ]
    )

    compatibility_normalized = minmax(
        [
            row[
                "field_abs_shap_compat_gain"
            ]
            for row in rows
        ]
    )

    for row_index, row in enumerate(rows):
        row["resource_gain_norm"] = float(
            resource_normalized[row_index]
        )

        row["compatibility_norm"] = float(
            compatibility_normalized[
                row_index
            ]
        )

    ranked_by_resource = sorted(
        rows,
        key=lambda row: (
            -row["resource_gain_norm"],
            row["cluster_set_str"],
        ),
    )

    for rank, row in enumerate(
        ranked_by_resource,
        start=1,
    ):
        row["resource_rank"] = rank

    pool_size = int(
        math.ceil(
            1.0
            + lambda_value
            * max(
                len(rows) - 1,
                0,
            )
        )
    )

    pool_size = max(
        1,
        min(
            pool_size,
            len(rows),
        ),
    )

    allowed = [
        row
        for row in rows
        if row["resource_rank"]
        <= pool_size
    ]

    selected = sorted(
        allowed,
        key=lambda row: (
            -row["compatibility_norm"],
            -row["resource_gain_norm"],
            row["cluster_set_str"],
        ),
    )[0]

    selected.update(
        {
            "lambda": float(
                lambda_value
            ),
            "resource_pool_size": (
                pool_size
            ),
            "num_candidate_cluster_sets": (
                len(rows)
            ),
        }
    )

    return selected


def select_top_k_for_cluster(
    cluster: Sequence[int],
    stl_shap: Dict[
        int,
        Dict[int, float],
    ],
    compact_feature_indices: Sequence[int],
    k: int,
) -> List[int]:
    cluster = canonical_cluster(cluster)

    ranked = sorted(
        (
            (
                int(index),
                float(
                    np.mean(
                        [
                            stl_shap[task].get(
                                int(index),
                                0.0,
                            )
                            for task in cluster
                        ]
                    )
                ),
            )
            for index
            in compact_feature_indices
        ),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    selected = [
        index
        for index, _
        in ranked[
            :min(
                max(k, 1),
                len(ranked),
            )
        ]
    ]

    return sorted(selected)


def select_low_k_for_mtl_all(
    active_tasks: Sequence[int],
    stl_shap: Dict[
        int,
        Dict[int, float],
    ],
    compact_feature_indices: Sequence[int],
    k: int,
) -> List[int]:
    """
    Select the K least-important compact bits using
    mean STL SHAP importance across all tasks.
    """

    ranked = sorted(
        (
            (
                int(index),
                float(
                    np.mean(
                        [
                            stl_shap[task].get(
                                int(index),
                                0.0,
                            )
                            for task
                            in active_tasks
                        ]
                    )
                ),
            )
            for index
            in compact_feature_indices
        ),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    selected = [
        index
        for index, _
        in ranked[
            :min(
                max(k, 1),
                len(ranked),
            )
        ]
    ]

    return sorted(selected)


def classes_by_output(
    model: Any,
) -> List[np.ndarray]:
    number_of_outputs = int(
        getattr(
            model,
            "n_outputs_",
            1,
        )
    )

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if number_of_outputs == 1:
        if isinstance(classes, list):
            return [
                np.asarray(classes[0])
            ]

        return [
            np.asarray(classes)
        ]

    return [
        np.asarray(output_classes)
        for output_classes in classes
    ]


def leaf_prediction(
    model: Any,
    leaf_id: int,
    tasks: Sequence[int],
) -> Dict[int, int]:
    leaf_values = (
        model.tree_.value[
            leaf_id
        ]
    )

    classes = classes_by_output(
        model
    )

    prediction: Dict[
        int,
        int,
    ] = {}

    for output_index, task in enumerate(tasks):
        values = np.asarray(
            leaf_values[output_index]
        )[
            :len(
                classes[output_index]
            )
        ]

        class_position = int(
            np.argmax(values)
        )

        prediction[int(task)] = int(
            classes[output_index][
                class_position
            ]
        )

    return prediction


def branch_constraint(
    threshold: float,
    branch: str,
):
    allowed = [
        bit
        for bit in (0, 1)
        if (
            bit <= threshold
            if branch == "left"
            else bit > threshold
        )
    ]

    if allowed == [0]:
        return 0

    if allowed == [1]:
        return 1

    if allowed == [0, 1]:
        return None

    return "impossible"


def bitstr_to_hex(bit_string: str) -> str:
    hexadecimal_width = int(
        math.ceil(len(bit_string) / 4.0)
    )

    return f"0x{int(bit_string, 2):0{hexadecimal_width}x}"

def export_tree_rules(
    model: Any,
    tasks: Sequence[int],
    suffix: str,
    key_width: int,
) -> List[Dict[str, Any]]:
    tree = model.tree_

    rules: List[
        Dict[str, Any]
    ] = []

    def recurse(
        node_id: int,
        constraints: Dict[int, int],
    ) -> None:
        left_child = int(
            tree.children_left[node_id]
        )

        right_child = int(
            tree.children_right[node_id]
        )

        is_leaf = (
            left_child == right_child
            or left_child < 0
        )

        if is_leaf:
            value_bits = [
                "0"
            ] * key_width

            mask_bits = [
                "0"
            ] * key_width

            for local_index, value in constraints.items():
                value_bits[local_index] = (
                    "1"
                    if value
                    else "0"
                )

                mask_bits[local_index] = "1"

            key_bits = "".join(
                value_bits
            )

            mask_string = "".join(
                mask_bits
            )

            prediction = leaf_prediction(
                model=model,
                leaf_id=node_id,
                tasks=tasks,
            )

            rules.append(
                {
                    "table": (
                        f"tb_hwimo_tree_"
                        f"{suffix}"
                    ),
                    "action": (
                        f"set_pred_"
                        f"{suffix}"
                    ),
                    "leaf_id": int(
                        node_id
                    ),
                    "key_bits": key_bits,
                    "mask_bits": (
                        mask_string
                    ),
                    "key_hex": (
                        bitstr_to_hex(
                            key_bits
                        )
                    ),
                    "mask_hex": (
                        bitstr_to_hex(
                            mask_string
                        )
                    ),
                    "priority": 1,
                    "pred": {
                        f"task{task}": int(
                            prediction[task]
                        )
                        for task in tasks
                    },
                    "params": [
                        int(
                            prediction[task]
                        )
                        for task in tasks
                    ],
                }
            )

            return

        feature = int(
            tree.feature[node_id]
        )

        threshold = float(
            tree.threshold[node_id]
        )

        for branch, child in (
            ("left", left_child),
            ("right", right_child),
        ):
            constraint = branch_constraint(
                threshold=threshold,
                branch=branch,
            )

            if constraint == "impossible":
                continue

            next_constraints = dict(
                constraints
            )

            if constraint is not None:
                old_constraint = (
                    next_constraints.get(
                        feature
                    )
                )

                if (
                    old_constraint is not None
                    and old_constraint
                    != constraint
                ):
                    continue

                next_constraints[
                    feature
                ] = int(constraint)

            recurse(
                node_id=child,
                constraints=next_constraints,
            )

    recurse(
        node_id=0,
        constraints={},
    )

    return rules


def rule_predict_one(
    key_bits: str,
    rules: Sequence[Dict[str, Any]],
    tasks: Sequence[int],
) -> Dict[int, int]:
    key_value = int(
        key_bits,
        2,
    )

    for rule in rules:
        mask_value = int(
            rule["mask_bits"],
            2,
        )

        rule_value = int(
            rule["key_bits"],
            2,
        )

        if (
            key_value
            & mask_value
        ) == (
            rule_value
            & mask_value
        ):
            return {
                int(task): int(value)
                for task, value
                in zip(
                    tasks,
                    rule["params"],
                )
            }

    raise RuntimeError(
        "no ternary rule matched"
    )


def verify_rules(
    model: DecisionTreeClassifier,
    X_test: np.ndarray,
    rules: Sequence[Dict[str, Any]],
    tasks: Sequence[int],
) -> None:
    model_prediction = np.asarray(
        model.predict(X_test)
    )

    if model_prediction.ndim == 1:
        model_prediction = (
            model_prediction.reshape(
                -1,
                1,
            )
        )

    for row_index, row in enumerate(X_test):
        key_bits = "".join(
            "1" if int(value) else "0"
            for value in row.tolist()
        )

        rule_prediction = (
            rule_predict_one(
                key_bits=key_bits,
                rules=rules,
                tasks=tasks,
            )
        )

        for output_index, task in enumerate(tasks):
            model_value = int(
                model_prediction[
                    row_index,
                    output_index,
                ]
            )

            rule_value = int(
                rule_prediction[
                    int(task)
                ]
            )

            if model_value != rule_value:
                raise RuntimeError(
                    "rule verification failed: "
                    f"row={row_index}, "
                    f"task={task}, "
                    f"model={model_value}, "
                    f"rule={rule_value}"
                )


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_selected_feature_files(
    cluster_dir: Path,
    selected_compact: np.ndarray,
    selected_global: np.ndarray,
    feature_k: int,
) -> Dict[str, Path]:
    npy_path = (
        cluster_dir
        / "selected_feature_indices.npy"
    )

    json_path = (
        cluster_dir
        / "selected_feature_indices.json"
    )

    csv_path = (
        cluster_dir
        / "selected_feature_indices.csv"
    )

    # 기존 pipeline에 맞춰 global 169-bit index 저장
    np.save(
        npy_path,
        selected_global,
    )

    write_json(
        json_path,
        {
            "index_space": (
                "global_169"
            ),
            "feature_k": int(
                feature_k
            ),
            "selected_feature_indices": (
                selected_global.tolist()
            ),
            "selected_compact_indices": (
                selected_compact.tolist()
            ),
        },
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "rank,"
            "global_index,"
            "compact_index,"
            "field\n"
        )

        for rank, (
            global_index,
            compact_index,
        ) in enumerate(
            zip(
                selected_global.tolist(),
                selected_compact.tolist(),
            ),
            start=1,
        ):
            field = BIT_TO_FIELD[
                int(compact_index)
            ]

            file.write(
                f"{rank},"
                f"{global_index},"
                f"{compact_index},"
                f"{field}\n"
            )

    return {
        "npy": npy_path,
        "json": json_path,
        "csv": csv_path,
    }


def train_export_cluster(
    cluster: Sequence[int],
    selected_compact_indices: Sequence[int],
    all_bitstrings: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: Dict[int, Dict[str, np.ndarray]],
    seed: int,
    depth: int,
    output_root: Path,
    feature_k: int,
    selection_name: str,
) -> Dict[str, Any]:
    cluster = canonical_cluster(
        cluster
    )

    suffix = "_".join(
        str(task)
        for task in cluster
    )

    cluster_dir = (
        output_root
        / cluster_slug(cluster)
    )

    cluster_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_all = bitstrings_to_numpy(
        bitstrings=all_bitstrings,
        selected_indices=(
            selected_compact_indices
        ),
    )

    X_train = X_all[
        train_idx
    ]

    X_test = X_all[
        test_idx
    ]

    Y_train = np.column_stack(
        [
            labels[task]["train"]
            for task in cluster
        ]
    )

    Y_test = np.column_stack(
        [
            labels[task]["test"]
            for task in cluster
        ]
    )

    model = make_tree(
        seed=seed,
        depth=depth,
    )

    model.fit(
        X_train,
        Y_train,
    )

    Y_prediction = np.asarray(
        model.predict(X_test)
    )

    if Y_prediction.ndim == 1:
        Y_prediction = (
            Y_prediction.reshape(
                -1,
                1,
            )
        )

    task_accuracy = {
        int(task): float(
            accuracy_score(
                Y_test[
                    :,
                    output_index,
                ],
                Y_prediction[
                    :,
                    output_index,
                ],
            )
            * 100.0
        )
        for output_index, task
        in enumerate(cluster)
    }

    selected_compact = np.asarray(
        selected_compact_indices,
        dtype=np.int64,
    )

    selected_global = np.asarray(
        [
            COMPACT_TO_GLOBAL[
                int(compact_index)
            ]
            for compact_index
            in selected_compact_indices
        ],
        dtype=np.int64,
    )

    model_path = (
        cluster_dir
        / (
            "model_mtl_feature_selected_"
            f"{selection_name}_"
            f"{feature_k}.pkl"
        )
    )

    results_path = (
        cluster_dir
        / "results.json"
    )

    rules_path = (
        cluster_dir
        / "rules.json"
    )

    with model_path.open(
        "wb"
    ) as file:
        pickle.dump(
            model,
            file,
        )

    selected_paths = (
        write_selected_feature_files(
            cluster_dir=cluster_dir,
            selected_compact=(
                selected_compact
            ),
            selected_global=(
                selected_global
            ),
            feature_k=feature_k,
        )
    )

    rules = export_tree_rules(
        model=model,
        tasks=cluster,
        suffix=suffix,
        key_width=len(
            selected_compact_indices
        ),
    )

    verify_rules(
        model=model,
        X_test=X_test,
        rules=rules,
        tasks=cluster,
    )

    write_json(
        rules_path,
        {
            "cluster": list(cluster),
            "cluster_str": (
                cluster_to_str(cluster)
            ),
            "tasks": [
                TASK_NAME_BY_NUM[task]
                for task in cluster
            ],
            "table": (
                f"tb_hwimo_tree_"
                f"{suffix}"
            ),
            "action": (
                f"set_pred_"
                f"{suffix}"
            ),
            "key_width": len(
                selected_compact_indices
            ),
            "selected_compact_indices": (
                selected_compact.tolist()
            ),
            "selected_global_indices": (
                selected_global.tolist()
            ),
            "num_entries": len(rules),
            "entries": rules,
        },
    )

    results = {
        "cluster": list(cluster),
        "cluster_str": (
            cluster_to_str(cluster)
        ),
        "task_names": [
            TASK_NAME_BY_NUM[task]
            for task in cluster
        ],
        "feature_selection": (
            selection_name
        ),
        "feature_k": int(
            feature_k
        ),
        "depth_limit": int(
            depth
        ),
        "selected_feature_indices_are": (
            "global_169_bit_indices"
        ),
        "selected_feature_indices": (
            selected_global.tolist()
        ),
        "selected_compact_indices": (
            selected_compact.tolist()
        ),
        "task_accuracy_percent": (
            task_accuracy
        ),
        "average_accuracy_percent": float(
            np.mean(
                list(
                    task_accuracy.values()
                )
            )
        ),
        "tree_node_count": int(
            model.tree_.node_count
        ),
        "tree_depth": int(
            model.tree_.max_depth
        ),
        "rule_count": int(
            len(rules)
        ),
        "rule_self_check": "passed",
        "files": {
            "model": str(
                model_path
            ),
            "selected_feature_indices_npy": str(
                selected_paths["npy"]
            ),
            "selected_feature_indices_json": str(
                selected_paths["json"]
            ),
            "selected_feature_indices_csv": str(
                selected_paths["csv"]
            ),
            "rules": str(
                rules_path
            ),
            "results": str(
                results_path
            ),
        },
    }

    write_json(
        results_path,
        results,
    )

    return {
        **results,
        "suffix": suffix,
        "rules": rules,
        "model_dir": str(
            cluster_dir
        ),
    }


def commands_for_result(
    result: Dict[str, Any],
    heading: str,
) -> List[str]:
    suffix = result["suffix"]

    table_name = (
        "MyIngress."
        f"tb_hwimo_tree_{suffix}"
    )

    action_name = (
        "MyIngress."
        f"set_pred_{suffix}"
    )

    lines = [
        heading,
        (
            "# cluster "
            f"{result['cluster_str']}"
        ),
    ]

    for rule in result["rules"]:
        parameters = " ".join(
            str(value)
            for value in rule["params"]
        )

        lines.append(
            f"table_add "
            f"{table_name} "
            f"{action_name} "
            f"{rule['key_hex']}"
            f"&&&"
            f"{rule['mask_hex']} "
            f"=> "
            f"{parameters} "
            f"{int(rule['priority'])}"
        )

    lines.append("")

    return lines


def collect_task_accuracies(
    results: Sequence[Dict[str, Any]],
    active_tasks: Sequence[int],
) -> Tuple[
    Dict[int, float],
    float,
]:
    task_accuracy_by_id: Dict[
        int,
        float,
    ] = {}

    for result in results:
        for task_id, accuracy in (
            result[
                "task_accuracy_percent"
            ].items()
        ):
            task_accuracy_by_id[
                int(task_id)
            ] = float(accuracy)

    missing_tasks = (
        set(active_tasks)
        - set(task_accuracy_by_id)
    )

    if missing_tasks:
        raise RuntimeError(
            "Missing task accuracy for: "
            f"{sorted(missing_tasks)}"
        )

    overall_average = float(
        np.mean(
            [
                task_accuracy_by_id[task]
                for task in active_tasks
            ]
        )
    )

    return (
        task_accuracy_by_id,
        overall_average,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_root(),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--packet-file",
        default="153_input.txt",
    )

    parser.add_argument(
        "--tasks",
        default="1,2,3,4,5,6,7",
    )

    parser.add_argument(
        "--feature-k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--cluster-mean-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--singleton-policy",
        choices=[
            "include_one",
            "include_zero",
            "exclude",
        ],
        default="include_one",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    parser.add_argument(
        "--stratify-task",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.lambda_value
        <= 1.0
    ):
        raise ValueError(
            "--lambda-value must be in [0, 1]"
        )

    if not (
        0.0
        <= args.cluster_mean_weight
        <= 1.0
    ):
        raise ValueError(
            "--cluster-mean-weight must be in [0, 1]"
        )

    if args.feature_k <= 0:
        raise ValueError(
            "--feature-k must be positive"
        )

    if args.depth <= 0:
        raise ValueError(
            "--depth must be positive"
        )

    if not (
        0.0
        < args.test_size
        < 1.0
    ):
        raise ValueError(
            "--test-size must be between 0 and 1"
        )

    base_dir = (
        args.base_dir.resolve()
    )

    data_dir = (
        args.data_dir.resolve()
        if args.data_dir
        else base_dir / "data"
    )

    proposed_models_root = (
        base_dir
        / "models"
        / f"top_{args.feature_k:03d}"
    )

    low_models_root = (
        base_dir
        / "models"
        / f"low_{args.feature_k:03d}"
    )

    proposed_models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    low_models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    commands_dir = (
        base_dir
        / "commands"
    )

    commands_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    proposed_commands_output = (
        commands_dir
        / (
            f"Hwimo_top"
            f"{args.feature_k:03d}"
            f"_proposed_commands.txt"
        )
    )

    low_commands_output = (
        commands_dir
        / (
            f"Hwimo_low"
            f"{args.feature_k:03d}"
            f"_mtl_all_commands.txt"
        )
    )

    active_tasks = parse_task_set(
        args.tasks
    )

    packet_path = (
        data_dir
        / args.packet_file
    )

    all_bitstrings = read_bitstrings(
        packet_path
    )

    number_of_samples = len(
        all_bitstrings
    )

    all_indices = np.arange(
        number_of_samples
    )

    all_labels: Dict[
        int,
        np.ndarray,
    ] = {}

    for task in active_tasks:
        label_path = (
            data_dir
            / f"task{task}_label_i.txt"
        )

        labels_for_task = read_label_file(
            label_path
        )

        if (
            len(labels_for_task)
            != number_of_samples
        ):
            raise ValueError(
                f"{label_path}: "
                f"label rows="
                f"{len(labels_for_task)}, "
                f"expected "
                f"{number_of_samples}"
            )

        all_labels[task] = (
            labels_for_task
        )

    if args.shuffle:
        stratify = None

        if (
            args.stratify_task
            is not None
        ):
            if (
                args.stratify_task
                not in active_tasks
            ):
                raise ValueError(
                    "--stratify-task "
                    f"{args.stratify_task} "
                    "is not included in "
                    f"tasks={active_tasks}"
                )

            stratify = all_labels[
                args.stratify_task
            ]

        train_idx, test_idx = (
            train_test_split(
                all_indices,
                test_size=args.test_size,
                random_state=args.seed,
                shuffle=True,
                stratify=stratify,
            )
        )

        train_idx = np.sort(
            train_idx
        )

        test_idx = np.sort(
            test_idx
        )

    else:
        split_point = int(
            number_of_samples
            * (
                1.0
                - args.test_size
            )
        )

        train_idx = all_indices[
            :split_point
        ]

        test_idx = all_indices[
            split_point:
        ]

    labels = {
        task: {
            "train": all_labels[task][
                train_idx
            ],
            "test": all_labels[task][
                test_idx
            ],
        }
        for task in active_tasks
    }

    own_feature_indices = {
        task: mask_to_indices(
            MASKS[task]
        )
        for task in active_tasks
    }

    compact_same_mask_indices = sorted(
        set(
            chain.from_iterable(
                own_feature_indices[task]
                for task in active_tasks
            )
        )
    )

    X_all_stl = bitstrings_to_numpy(
        bitstrings=all_bitstrings,
        selected_indices=(
            compact_same_mask_indices
        ),
    )
    print("=" * 70)
    print("parameters")
    print("=" * 70)
    print(f"Feature K: {args.feature_k}")
    print(f"Depth: {args.depth}")
    print(f"Lambda: {args.lambda_value:.2f}")
    print("="*70)

    print(
        "\n[1/4] Train STL models "
        "and compute proxies"
    )

    (
        stl_shap,
        stl_used,
        stl_vectors,
    ) = train_single_task_trees(
        active_tasks=active_tasks,
        X_train=X_all_stl[
            train_idx
        ],
        X_test=X_all_stl[
            test_idx
        ],
        labels=labels,
        compact_same_mask_indices=(
            compact_same_mask_indices
        ),
        seed=args.seed,
        depth=args.depth,
    )

    print(
        "\n[2/4] Select Proposed "
        "cluster set"
    )

    selected = select_cluster_set(
        active_tasks=active_tasks,
        used=stl_used,
        vectors=stl_vectors,
        lambda_value=(
            args.lambda_value
        ),
        mean_weight=(
            args.cluster_mean_weight
        ),
        singleton_policy=(
            args.singleton_policy
        ),
    )

    print(
        "selected:",
        selected[
            "cluster_set_str"
        ],
    )
    print()

    print(
        "[3/4] Train Proposed "
        "clusters and export rules"
    )


    proposed_results: List[
        Dict[str, Any]
    ] = []

    proposed_command_lines = [
        (
            "# BMv2 simple_switch_CLI "
            "commands for Proposed"
        ),
        "",
    ]

    selected_clusters = list(
        selected["cluster_set"]
    )

    for cluster in tqdm(
        selected_clusters,
        desc="Proposed clusters",
        unit="cluster",
        dynamic_ncols=True,
    ):
        selected_indices = (
            select_top_k_for_cluster(
                cluster=cluster,
                stl_shap=stl_shap,
                compact_feature_indices=(
                    compact_same_mask_indices
                ),
                k=args.feature_k,
            )
        )

        result = train_export_cluster(
            cluster=cluster,
            selected_compact_indices=(
                selected_indices
            ),
            all_bitstrings=(
                all_bitstrings
            ),
            train_idx=train_idx,
            test_idx=test_idx,
            labels=labels,
            seed=args.seed,
            depth=args.depth,
            output_root=(
                proposed_models_root
            ),
            feature_k=args.feature_k,
            selection_name="top",
        )

        proposed_results.append(
            result
        )

        proposed_command_lines.extend(
            commands_for_result(
                result=result,
                heading="# Proposed",
            )
        )

    proposed_commands_output.write_text(
        "\n".join(
            proposed_command_lines
        ),
        encoding="utf-8",
    )

    (
        proposed_task_accuracy,
        proposed_overall_accuracy,
    ) = collect_task_accuracies(
        results=proposed_results,
        active_tasks=active_tasks,
    )

    proposed_summary = {
        "program": "proposed",
        "selected_cluster_set": (
            selected[
                "cluster_set_str"
            ]
        ),
        "selection_mode": (
            "ResourcePool -> "
            "Field-AbsSHAP"
        ),
        "lambda": float(
            args.lambda_value
        ),
        "cluster_mean_weight": float(
            args.cluster_mean_weight
        ),
        "singleton_policy": (
            args.singleton_policy
        ),
        "resource_pool_size": int(
            selected[
                "resource_pool_size"
            ]
        ),
        "num_candidate_cluster_sets": int(
            selected[
                "num_candidate_cluster_sets"
            ]
        ),
        "resource_gain_saved_features": float(
            selected[
                "resource_gain_saved_features"
            ]
        ),
        "field_abs_shap_compat_gain": float(
            selected[
                "field_abs_shap_compat_gain"
            ]
        ),
        "feature_k": int(
            args.feature_k
        ),
        "depth": int(
            args.depth
        ),
        "task_accuracy_percent": {
            str(task): (
                proposed_task_accuracy[
                    task
                ]
            )
            for task in active_tasks
        },
        "overall_average_accuracy_percent": (
            proposed_overall_accuracy
        ),
        "clusters": [
            {
                "cluster": result[
                    "cluster"
                ],
                "cluster_str": result[
                    "cluster_str"
                ],
                "model_dir": result[
                    "model_dir"
                ],
                "rule_count": int(
                    result[
                        "rule_count"
                    ]
                ),
            }
            for result
            in proposed_results
        ],
        "commands_file": str(
            proposed_commands_output
        ),
    }

    write_json(
        proposed_models_root
        / "selected_cluster_set.json",
        proposed_summary,
    )

    print(
        "\n[4/4] Train MTL-all "
        "and export rules"
    )

    mtl_all_cluster = tuple(
        active_tasks
    )

    low_selected_indices = (
        select_low_k_for_mtl_all(
            active_tasks=active_tasks,
            stl_shap=stl_shap,
            compact_feature_indices=(
                compact_same_mask_indices
            ),
            k=args.feature_k,
        )
    )

    with tqdm(
        total=1,
        desc="MTL-all",
        unit="cluster",
        dynamic_ncols=True,
    ) as progress:
        low_result = train_export_cluster(
            cluster=mtl_all_cluster,
            selected_compact_indices=low_selected_indices,
            all_bitstrings=all_bitstrings,
            train_idx=train_idx,
            test_idx=test_idx,
            labels=labels,
            seed=args.seed,
            depth=args.depth,
            output_root=low_models_root,
            feature_k=args.feature_k,
            selection_name="low",
        )

        progress.update(1)

    low_command_lines = [
        (
            "# BMv2 simple_switch_CLI "
            "commands for MTL-all"
        ),
        "",
    ]

    low_command_lines.extend(
        commands_for_result(
            result=low_result,
            heading=(
                "# MTL-all"
            ),
        )
    )

    low_commands_output.write_text(
        "\n".join(
            low_command_lines
        ),
        encoding="utf-8",
    )

    (
        low_task_accuracy,
        low_overall_accuracy,
    ) = collect_task_accuracies(
        results=[low_result],
        active_tasks=active_tasks,
    )

    low_summary = {
        "program": "mtl_all_low",
        "cluster": list(
            mtl_all_cluster
        ),
        "cluster_str": (
            cluster_to_str(
                mtl_all_cluster
            )
        ),
        "feature_selection": "low",
        "feature_k": int(
            args.feature_k
        ),
        "depth": int(
            args.depth
        ),
        "task_accuracy_percent": {
            str(task): (
                low_task_accuracy[
                    task
                ]
            )
            for task in active_tasks
        },
        "overall_average_accuracy_percent": (
            low_overall_accuracy
        ),
        "model_dir": low_result[
            "model_dir"
        ],
        "rule_count": int(
            low_result[
                "rule_count"
            ]
        ),
        "commands_file": str(
            low_commands_output
        ),
    }

    write_json(
        low_models_root
        / "mtl_all_low_summary.json",
        low_summary,
    )

    print()
    print("=" * 70)

    # print(
    #     "Proposed overall average accuracy: "
    #     f"{proposed_overall_accuracy:.2f}%"
    # )

    # print(
    #     "MTL-all Low-K overall average accuracy: "
    #     f"{low_overall_accuracy:.2f}%"
    # )

    # print()
    print(
        "Proposed rule commands: "
        f"{proposed_commands_output}"
    )

    print(
        "MTL-all rule commands: "
        f"{low_commands_output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()