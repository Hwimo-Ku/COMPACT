#!/usr/bin/env python3
"""
cluster_set_selection.py

HWIMO COMPACT 153-bit cluster-set selector.

Pipeline:
  1. Train single-task trees.
  2. Compute STL mean_abs SHAP and STL used split-node feature sets.
  3. Generate all candidate cluster sets.
  4. Select one cluster set using ResourcePool -> Field-AbsSHAP compatibility.
     Default lambda = 0.6.
  5. For the selected cluster set only:
       - select Top-K features per cluster by cluster-mean STL mean_abs SHAP
       - train feature-selected MTL trees
  6. Print only selected cluster set and task accuracies.

Default input:
  $ROOT/data/153_input.txt
  $ROOT/data/task1_label_i.txt
  ...
  $ROOT/data/task7_label_i.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import warnings
from collections import deque
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

if "bool" not in np.__dict__:
    setattr(np, "bool", np.bool_)
if "int" not in np.__dict__:
    setattr(np, "int", int)

warnings.filterwarnings("ignore", message="In the future `np.bool`")

try:
    import shap
except ModuleNotFoundError:
    raise SystemExit(
        "[ERROR] Python package 'shap' is not installed.\n"
        "Install it on the server with:\n"
        "  python3 -m pip install --user shap\n"
    )

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

BIT_TO_FIELD_169: Dict[int, str] = {}
for field_name, start in FEATURE_START_169.items():
    for bit_idx in range(start, start + FEATURE_LENGTHS_169[field_name]):
        BIT_TO_FIELD_169[bit_idx] = field_name

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
    + list(range(REMOVED_GLOBAL_END_EXCL, GLOBAL_WIDTH))
)

GLOBAL_TO_COMPACT = {
    global_idx: compact_idx
    for compact_idx, global_idx in enumerate(COMPACT_TO_GLOBAL)
}

BIT_TO_FIELD: Dict[int, str] = {
    compact_idx: BIT_TO_FIELD_169[global_idx]
    for compact_idx, global_idx in enumerate(COMPACT_TO_GLOBAL)
}


MASKS_169 = {
    1: "0" * 16 + "0" * 8 + "1" * 32 + "1" * 32 + "0" * 16 + "1" * 16 + "0" * 6 + "0" * 16 + "1" * 8 + "1" * 3 + "1" * 16,
    2: "1" * 16 + "0" * 8 + "1" * 32 + "1" * 32 + "1" * 16 + "1" * 16 + "0" * 6 + "0" * 16 + "0" * 8 + "0" * 3 + "1" * 16,
    3: "0" * 16 + "0" * 8 + "1" * 32 + "0" * 32 + "1" * 16 + "1" * 16 + "0" * 6 + "0" * 16 + "1" * 8 + "1" * 3 + "1" * 16,
    4: "0" * 16 + "0" * 8 + "1" * 32 + "0" * 32 + "1" * 16 + "1" * 16 + "0" * 6 + "0" * 16 + "1" * 8 + "1" * 3 + "1" * 16,
    5: "1" * 16 + "1" * 8 + "0" * 32 + "0" * 32 + "0" * 16 + "1" * 16 + "1" * 6 + "0" * 16 + "0" * 8 + "1" * 3 + "1" * 16,
    6: "1" * 16 + "1" * 8 + "0" * 32 + "1" * 32 + "1" * 16 + "0" * 16 + "1" * 6 + "0" * 16 + "0" * 8 + "0" * 3 + "1" * 16,
    7: "1" * 16 + "0" * 8 + "1" * 32 + "1" * 32 + "0" * 16 + "0" * 16 + "0" * 6 + "0" * 16 + "1" * 8 + "1" * 3 + "1" * 16,
}


def compact_mask_from_169(mask_169: str) -> str:
    if len(mask_169) != GLOBAL_WIDTH:
        raise ValueError(f"expected 169-bit mask, got {len(mask_169)}")
    return "".join(mask_169[gidx] for gidx in COMPACT_TO_GLOBAL)


MASKS = {
    task_id: compact_mask_from_169(mask)
    for task_id, mask in MASKS_169.items()
}


def default_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return here.parent


def canonical_cluster(cluster: Iterable[int]) -> Tuple[int, ...]:
    out = tuple(sorted(int(x) for x in cluster))
    for task_id in out:
        if task_id not in TASK_NAME_BY_NUM:
            raise ValueError(f"unknown task id: {task_id}")
    return out


def canonical_cluster_set(cluster_set: Iterable[Iterable[int]]) -> Tuple[Tuple[int, ...], ...]:
    clusters = [canonical_cluster(c) for c in cluster_set]
    return tuple(sorted(clusters, key=lambda c: (len(c), c)))


def cluster_to_str(cluster: Iterable[int]) -> str:
    c = canonical_cluster(cluster)
    return "{" + ",".join(str(x) for x in c) + "}"


def cluster_set_to_str(cluster_set: Iterable[Iterable[int]]) -> str:
    return " | ".join(cluster_to_str(c) for c in canonical_cluster_set(cluster_set))


def parse_task_set(s: str) -> List[int]:
    s = s.strip()
    if not s:
        raise ValueError("empty task set")

    s = s.replace("{", "").replace("}", "")
    tasks = [int(x.strip()) for x in s.split(",") if x.strip()]

    if len(tasks) != len(set(tasks)):
        raise ValueError(f"duplicated task in task set: {s}")

    for t in tasks:
        if t not in TASK_NAME_BY_NUM:
            raise ValueError(f"unknown task id: {t}")

    return sorted(tasks)


def task_names(cluster: Iterable[int]) -> str:
    return ",".join(TASK_NAME_BY_NUM[t] for t in canonical_cluster(cluster))


def mask_to_indices(mask_str: str) -> List[int]:
    return [idx for idx, bit in enumerate(mask_str) if bit == "1"]


def read_bitstrings(path: Path, width: int = COMPACT_WIDTH) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = [line.strip().split()[0] for line in f if line.strip()]

    widths = sorted(set(len(x) for x in rows))
    if widths != [width]:
        raise ValueError(f"{path}: expected {width}-bit rows, observed widths={widths}")

    for i, row in enumerate(rows[:1000]):
        if any(ch not in "01" for ch in row):
            raise ValueError(f"{path}: non-binary value observed near row {i}")

    return rows


def read_label_file(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return np.asarray(
            [int(line.strip().split()[0]) for line in f if line.strip()],
            dtype=np.int64,
        )


def bitstrings_to_numpy(bitstrings: Sequence[str], selected_indices: Sequence[int]) -> np.ndarray:
    selected_indices = [int(i) for i in selected_indices]
    return np.asarray(
        [[int(row[i]) for i in selected_indices] for row in bitstrings],
        dtype=np.int8,
    )


def check_required_training_files(data_dir: Path, packet_file: str, active_tasks: Sequence[int]) -> None:
    required = [data_dir / packet_file]
    required += [data_dir / f"task{int(t)}_label_i.txt" for t in active_tasks]

    missing = []

    for path in required:
        if path.exists() and path.is_file():
            try:
                n_lines = sum(1 for _ in open(path, "r", encoding="utf-8", errors="replace"))
            except Exception:
                n_lines = "unknown"
            # print(f"[OK]      {path}  lines={n_lines}")
        else:
            # print(f"[MISSING] {path}")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Required training input file(s) are missing:\n"
            + "\n".join(str(x) for x in missing)
        )


def set_partitions(items: Sequence[int]):
    items = tuple(items)

    if len(items) == 0:
        yield tuple()
        return

    first = items[0]

    for smaller in set_partitions(items[1:]):
        for i in range(len(smaller)):
            new_block = tuple(sorted((first,) + smaller[i]))
            new_cluster_set = list(smaller[:i]) + [new_block] + list(smaller[i + 1:])
            yield canonical_cluster_set(new_cluster_set)

        yield canonical_cluster_set(((first,),) + smaller)


def minmax(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_l1(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = float(np.sum(np.abs(v)))
    if s <= 1e-12:
        return np.zeros_like(v)
    return v / s


def make_tree(seed: int, depth: int | None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
    )


def get_used_split_features(model: DecisionTreeClassifier, compact_feature_indices: Sequence[int]) -> set[int]:
    tree_ = model.tree_
    used_local = np.unique(tree_.feature[tree_.feature >= 0]).astype(int)
    compact_feature_indices = np.asarray(compact_feature_indices, dtype=int)
    used_compact = compact_feature_indices[used_local]
    return set(int(x) for x in used_compact.tolist())


def convert_shap_values_to_sample_matrix(shap_values, pred_label: np.ndarray) -> np.ndarray:
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        n_samples, n_features = shap_values[0].shape
        out = np.zeros((n_samples, n_features), dtype=float)

        for i in range(n_samples):
            cls = int(pred_label[i])
            cls = max(0, min(cls, n_classes - 1))
            out[i] = shap_values[cls][i]

        return out

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr.astype(float)

    if arr.ndim == 3:
        n_samples = arr.shape[0]

        if arr.shape[1] > 1 and arr.shape[2] <= 100:
            out = np.zeros((n_samples, arr.shape[1]), dtype=float)

            for i in range(n_samples):
                cls = int(pred_label[i])
                cls = max(0, min(cls, arr.shape[2] - 1))
                out[i] = arr[i, :, cls]

            return out

        if arr.shape[1] <= 100:
            out = np.zeros((n_samples, arr.shape[2]), dtype=float)

            for i in range(n_samples):
                cls = int(pred_label[i])
                cls = max(0, min(cls, arr.shape[1] - 1))
                out[i] = arr[i, cls, :]

            return out

    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


def compute_mean_abs_shap(model: DecisionTreeClassifier, X_eval: np.ndarray) -> np.ndarray:
    pred_label = model.predict(X_eval)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    sample_shap = convert_shap_values_to_sample_matrix(shap_values, pred_label)
    return np.mean(np.abs(sample_shap), axis=0)


def compact_shap_to_field_vector(
    shap_by_compact_idx: Dict[int, float],
    compact_feature_indices: Sequence[int],
) -> np.ndarray:
    by_field = {field: 0.0 for field in FIELD_ORDER}

    for compact_idx in compact_feature_indices:
        field = BIT_TO_FIELD.get(int(compact_idx), "unknown")
        if field == "ip_id":
            continue
        if field not in by_field:
            by_field[field] = 0.0
        by_field[field] += float(shap_by_compact_idx.get(int(compact_idx), 0.0))

    return normalize_l1(np.asarray([by_field[f] for f in FIELD_ORDER], dtype=float))


def train_single_task_trees(
    active_tasks: Sequence[int],
    X_train_global: np.ndarray,
    X_test_global: np.ndarray,
    labels: Dict[int, Dict[str, np.ndarray]],
    compact_same_mask_indices: Sequence[int],
    seed: int,
    depth: int,
):
    stl_shap = {}
    stl_used_features = {}
    stl_field_vectors = {}

    print("\nTrain single-task trees")

    for task_id in active_tasks:
        model = make_tree(seed=seed, depth=depth)
        model.fit(X_train_global, labels[task_id]["train"])

        mean_abs_local = compute_mean_abs_shap(model, X_test_global)

        shap_by_compact_idx = {
            int(compact_same_mask_indices[local_idx]): float(score)
            for local_idx, score in enumerate(mean_abs_local)
        }

        stl_shap[task_id] = shap_by_compact_idx
        stl_used_features[task_id] = get_used_split_features(model, compact_same_mask_indices)
        stl_field_vectors[task_id] = compact_shap_to_field_vector(
            shap_by_compact_idx,
            compact_same_mask_indices,
        )

        print(f"  task{task_id:<2d} {TASK_NAME_BY_NUM[task_id]:<14s} done")

    return stl_shap, stl_used_features, stl_field_vectors


def cluster_resource_gain(cluster: Sequence[int], stl_used_features: Dict[int, set[int]]) -> float:
    cluster = canonical_cluster(cluster)
    separate_usage = sum(len(stl_used_features[t]) for t in cluster)
    union_usage = len(set().union(*(stl_used_features[t] for t in cluster)))
    return float(separate_usage - union_usage)


def cluster_compat_gain(
    cluster: Sequence[int],
    stl_field_vectors: Dict[int, np.ndarray],
    w_mean: float,
    singleton_policy: str,
) -> float:
    cluster = canonical_cluster(cluster)

    if len(cluster) == 1:
        if singleton_policy == "include_one":
            return 1.0
        if singleton_policy == "include_zero":
            return 0.0
        if singleton_policy == "exclude":
            return math.nan
        raise ValueError(f"unknown singleton policy: {singleton_policy}")

    vals = []
    for a, b in combinations(cluster, 2):
        vals.append(max(0.0, min(1.0, cosine(stl_field_vectors[a], stl_field_vectors[b]))))

    mean_v = float(np.mean(vals))
    min_v = float(np.min(vals))
    w_min = 1.0 - float(w_mean)

    return float(float(w_mean) * mean_v + w_min * min_v)


def score_cluster_set(
    cluster_set: Sequence[Sequence[int]],
    active_tasks: Sequence[int],
    stl_used_features: Dict[int, set[int]],
    stl_field_vectors: Dict[int, np.ndarray],
    w_mean: float,
    singleton_policy: str,
) -> Dict:
    cluster_set = canonical_cluster_set(cluster_set)

    resource_gain = sum(
        cluster_resource_gain(c, stl_used_features)
        for c in cluster_set
    )

    compat_num = 0.0
    compat_den = 0.0

    for c in cluster_set:
        g = cluster_compat_gain(
            c,
            stl_field_vectors=stl_field_vectors,
            w_mean=w_mean,
            singleton_policy=singleton_policy,
        )

        if math.isnan(g):
            continue

        weight = float(len(c))
        compat_num += weight * float(g)
        compat_den += weight

    if compat_den <= 1e-12:
        compat = 0.0
    else:
        compat = compat_num / compat_den

    return {
        "cluster_set": cluster_set,
        "cluster_set_str": cluster_set_to_str(cluster_set),
        "resource_gain_saved_features": float(resource_gain),
        "field_abs_shap_compat_gain": float(compat),
        "num_clusters": int(len(cluster_set)),
    }


def select_cluster_set(
    active_tasks: Sequence[int],
    stl_used_features: Dict[int, set[int]],
    stl_field_vectors: Dict[int, np.ndarray],
    lam: float,
    w_mean: float,
    singleton_policy: str,
) -> Dict:
    all_cluster_sets = sorted(
        set(set_partitions(active_tasks)),
        key=lambda cs: cluster_set_to_str(cs),
    )

    rows = [
        score_cluster_set(
            cs,
            active_tasks=active_tasks,
            stl_used_features=stl_used_features,
            stl_field_vectors=stl_field_vectors,
            w_mean=w_mean,
            singleton_policy=singleton_policy,
        )
        for cs in all_cluster_sets
    ]

    resource_norm = minmax([r["resource_gain_saved_features"] for r in rows])
    compat_norm = minmax([r["field_abs_shap_compat_gain"] for r in rows])

    for i, r in enumerate(rows):
        r["resource_gain_norm"] = float(resource_norm[i])
        r["field_abs_shap_compat_gain_norm"] = float(compat_norm[i])

    rows_sorted_by_resource = sorted(
        rows,
        key=lambda r: (-r["resource_gain_norm"], r["cluster_set_str"]),
    )

    for rank, r in enumerate(rows_sorted_by_resource, start=1):
        r["resource_rank"] = int(rank)

    n = len(rows)
    pool_size = int(math.ceil(1.0 + float(lam) * max(n - 1, 0)))
    pool_size = max(1, min(pool_size, n))

    allowed = [r for r in rows if int(r["resource_rank"]) <= pool_size]

    selected = sorted(
        allowed,
        key=lambda r: (
            -r["field_abs_shap_compat_gain_norm"],
            -r["resource_gain_norm"],
            r["cluster_set_str"],
        ),
    )[0]

    selected["lambda"] = float(lam)
    selected["resource_pool_size"] = int(pool_size)
    selected["num_candidate_cluster_sets"] = int(n)

    return selected


def select_top_k_for_cluster(
    cluster: Sequence[int],
    stl_shap: Dict[int, Dict[int, float]],
    compact_feature_indices: Sequence[int],
    k: int,
) -> List[int]:
    cluster = canonical_cluster(cluster)
    scores = {}

    for compact_idx in compact_feature_indices:
        vals = [stl_shap[t].get(int(compact_idx), 0.0) for t in cluster]
        scores[int(compact_idx)] = float(np.mean(vals))

    k = min(max(int(k), 1), len(compact_feature_indices))
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    return sorted(int(idx) for idx, _ in ranked[:k])


def train_selected_cluster(
    cluster: Sequence[int],
    selected_compact_indices: Sequence[int],
    all_bitstrings: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: Dict[int, Dict[str, np.ndarray]],
    seed: int,
    depth: int,
) -> Dict:
    cluster = canonical_cluster(cluster)

    X_all = bitstrings_to_numpy(all_bitstrings, selected_compact_indices)
    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    Y_train = np.column_stack([labels[t]["train"] for t in cluster])
    Y_test = np.column_stack([labels[t]["test"] for t in cluster])

    model = make_tree(seed=seed, depth=depth)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    task_accs = [
        float(accuracy_score(Y_test[:, i], Y_pred[:, i]) * 100.0)
        for i in range(Y_test.shape[1])
    ]

    return {
        "cluster": cluster,
        "task_accs": {
            int(task_id): float(task_accs[i])
            for i, task_id in enumerate(cluster)
        },
        "avg_acc": float(np.mean(task_accs)),
        "min_acc": float(np.min(task_accs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-dir", type=Path, default=default_root())
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--packet-file", default="153_input.txt")
    ap.add_argument("--tasks", default="1,2,3,4,5,6,7")
    ap.add_argument("--feature-k", type=int, default=30)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--lambda-value", type=float, default=0.6)
    ap.add_argument("--cluster-mean-weight", type=float, default=0.5)
    ap.add_argument("--singleton-policy", choices=["include_one", "include_zero", "exclude"], default="include_one")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--stratify-task", type=int, default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--summary-json", type=Path, default=None)

    args = ap.parse_args()

    base_dir = args.base_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else base_dir / "data"
    packet_path = data_dir / args.packet_file

    active_tasks = parse_task_set(args.tasks)

    check_required_training_files(
        data_dir=data_dir,
        packet_file=args.packet_file,
        active_tasks=active_tasks,
    )

    if args.check_only:
        print("CHECK ONLY: required training files exist.")
        return

    all_bitstrings = read_bitstrings(packet_path, width=COMPACT_WIDTH)
    n_samples = len(all_bitstrings)
    all_indices = np.arange(n_samples)

    label_all = {}

    for task_id in active_tasks:
        label_path = data_dir / f"task{task_id}_label_i.txt"
        y = read_label_file(label_path)

        if len(y) != n_samples:
            raise ValueError(f"{label_path}: label rows={len(y)}, expected {n_samples}")

        label_all[task_id] = y

    if args.shuffle:
        stratify = None

        if args.stratify_task is not None:
            if args.stratify_task not in active_tasks:
                raise ValueError(
                    f"--stratify-task {args.stratify_task} not in active tasks={active_tasks}"
                )

            stratify = label_all[args.stratify_task]

        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
            stratify=stratify,
        )

        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

    else:
        split_point = int(n_samples * (1.0 - args.test_size))
        train_idx = all_indices[:split_point]
        test_idx = all_indices[split_point:]

    labels = {
        t: {
            "train": label_all[t][train_idx],
            "test": label_all[t][test_idx],
        }
        for t in active_tasks
    }

    own_feature_indices = {
        t: mask_to_indices(MASKS[t])
        for t in active_tasks
    }

    compact_same_mask_indices = sorted(
        set(chain.from_iterable(own_feature_indices[t] for t in active_tasks))
    )

    X_all_global = bitstrings_to_numpy(all_bitstrings, compact_same_mask_indices)
    X_train_global = X_all_global[train_idx]
    X_test_global = X_all_global[test_idx]

    print("=" * 90)
    print("COMPACT cluster set selection")
    print("=" * 90)
    # print(f"BASE_DIR:          {base_dir}")
    # print(f"DATA_DIR:          {data_dir}")
    # print(f"packet_file:       {packet_path}")
    print(f"input width:       {COMPACT_WIDTH} bits")
    print(f"tasks:             {active_tasks}")
    print(f"feature_k:         {args.feature_k}")
    print(f"depth:             {args.depth}")
    print(f"lambda:            {args.lambda_value}")
    # print(f"samples:           total={n_samples}, train={len(train_idx)}, test={len(test_idx)}")
    # print(f"same-mask bits:    {len(compact_same_mask_indices)}")
    print("=" * 90)

    stl_shap, stl_used_features, stl_field_vectors = train_single_task_trees(
        active_tasks=active_tasks,
        X_train_global=X_train_global,
        X_test_global=X_test_global,
        labels=labels,
        compact_same_mask_indices=compact_same_mask_indices,
        seed=args.seed,
        depth=args.depth,
    )

    selected = select_cluster_set(
        active_tasks=active_tasks,
        stl_used_features=stl_used_features,
        stl_field_vectors=stl_field_vectors,
        lam=args.lambda_value,
        w_mean=args.cluster_mean_weight,
        singleton_policy=args.singleton_policy,
    )

    selected_cluster_set = selected["cluster_set"]

    print("\n" + "=" * 90)
    print("SELECTED CLUSTER SET")
    print("=" * 90)
    print(f"cluster set:       {selected['cluster_set_str']}")
    print(f"lambda:            {selected['lambda']}")
    print(f"candidate count:   {selected['num_candidate_cluster_sets']}")
    print(f"resource pool:     {selected['resource_pool_size']}")
    print(f"resource gain:     {selected['resource_gain_saved_features']:.6f}")
    print(f"compatibility gain:       {selected['field_abs_shap_compat_gain']:.6f}")
    print("=" * 90)

    cluster_results = []
    task_acc_rows = []

    for cluster in selected_cluster_set:
        selected_indices = select_top_k_for_cluster(
            cluster=cluster,
            stl_shap=stl_shap,
            compact_feature_indices=compact_same_mask_indices,
            k=args.feature_k,
        )

        result = train_selected_cluster(
            cluster=cluster,
            selected_compact_indices=selected_indices,
            all_bitstrings=all_bitstrings,
            train_idx=train_idx,
            test_idx=test_idx,
            labels=labels,
            seed=args.seed,
            depth=args.depth,
        )

        cluster_results.append(result)

        # print(
        #     f"cluster={cluster_to_str(cluster):<12s} "
        #     f"tasks={task_names(cluster):<40s} "
        #     f"avg_acc={result['avg_acc']:.4f}%"
        # )

        for task_id, acc in result["task_accs"].items():
            task_acc_rows.append(
                {
                    "cluster": cluster_to_str(cluster),
                    "task_id": task_id,
                    "task_name": TASK_NAME_BY_NUM[task_id],
                    "accuracy": acc,
                }
            )

    all_task_accs = [row["accuracy"] for row in task_acc_rows]
    cluster_set_avg = float(np.mean(all_task_accs))
    cluster_set_min = float(np.min(all_task_accs))

    # print("=" * 90)
    # print("CLUSTER SET RESULT")
    # print("=" * 90)
    # print(f"cluster set:      {selected['cluster_set_str']}")
    # print(f"feature_k:        {args.feature_k}")
    # print(f"avg accuracy:     {cluster_set_avg:.4f}%")
    # print("-" * 90)
    # print(f"{'Task':<8} {'Name':<16} {'Cluster':<14} {'Accuracy [%]':>14}")

    # for row in sorted(task_acc_rows, key=lambda x: x["task_id"]):
    #     print(
    #         f"{row['task_id']:<8} "
    #         f"{row['task_name']:<16} "
    #         f"{row['cluster']:<14} "
    #         f"{row['accuracy']:>14.4f}"
    #     )

    # print("=" * 90)

    if args.summary_json is not None:
        summary_path = args.summary_json

        if not summary_path.is_absolute():
            summary_path = base_dir / summary_path

        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "selected_cluster_set": selected["cluster_set_str"],
            "lambda": float(args.lambda_value),
            "feature_k": int(args.feature_k),
            "depth": int(args.depth),
            "selection_mode": "ResourcePool -> Field-AbsSHAP",
            "resource_pool_size": int(selected["resource_pool_size"]),
            "num_candidate_cluster_sets": int(selected["num_candidate_cluster_sets"]),
            "resource_gain_saved_features": float(selected["resource_gain_saved_features"]),
            "field_abs_shap_compat_gain": float(selected["field_abs_shap_compat_gain"]),
            "avg_accuracy": cluster_set_avg,
            "min_accuracy": cluster_set_min,
            "task_results": task_acc_rows,
        }

        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
