#!/usr/bin/env python3
"""
Generate BMv2 artifacts for HWIMO top_030 models.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

TASK_WIDTH = {1: 1, 2: 1, 3: 3, 4: 3, 5: 2, 6: 2, 7: 2}

PROGRAMS = {
    "Hwimo_top030_proposed": [
        {"cluster": "cluster_1", "suffix": "1", "tasks": [1]},
        {"cluster": "cluster_2", "suffix": "2", "tasks": [2]},
        {"cluster": "cluster_3", "suffix": "3", "tasks": [3]},
        {"cluster": "cluster_4", "suffix": "4", "tasks": [4]},
        {"cluster": "cluster_5_6_7", "suffix": "5_6_7", "tasks": [5, 6, 7]},
    ],
    "Hwimo_top030_mtl_all": [
        {"cluster": "cluster_1_2_3_4_5_6_7", "suffix": "1_2_3_4_5_6_7", "tasks": [1, 2, 3, 4, 5, 6, 7]},
    ],
}


def global_to_compact(g: int) -> int:
    if 0 <= g <= 125:
        return g
    if 126 <= g <= 141:
        raise ValueError(f"global index {g} is in removed ip_id range 126..141")
    if 142 <= g <= 168:
        return g - 16
    raise ValueError(f"global index {g} outside 0..168")


def compact_to_p4_bit(compact_idx: int) -> int:
    if not (0 <= compact_idx <= 152):
        raise ValueError(f"compact index outside 0..152: {compact_idx}")
    return 152 - compact_idx


def bitstr_to_hex(s: str) -> str:
    width = int(math.ceil(len(s) / 4.0))
    return f"0x{int(s, 2):0{width}x}"


def read_meta(path: Path) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def read_feature_lines(path: Path) -> List[str]:
    lines, widths = [], set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip().split()[0]
            if s:
                widths.add(len(s))
                lines.append(s)
    if widths != {153}:
        raise RuntimeError(f"{path} expected only 153-bit rows, observed widths={sorted(widths)}")
    return lines


def read_labels(root: Path, start_idx: int, n_test: int) -> Dict[int, List[int]]:
    labels = {}
    for t in range(1, 8):
        p = root / "data" / f"task{t}_label_i.txt"
        vals = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    vals.append(int(line.split()[0]))
        if len(vals) < start_idx + n_test:
            raise RuntimeError(f"{p} has {len(vals)} rows, need {start_idx+n_test}")
        labels[t] = vals[start_idx:start_idx+n_test]
    return labels


def classes_by_output(model: Any) -> List[np.ndarray]:
    n_out = int(getattr(model, "n_outputs_", 1))
    classes = getattr(model, "classes_", None)
    if n_out == 1:
        if isinstance(classes, list):
            return [np.asarray(classes[0])]
        return [np.asarray(classes)]
    return [np.asarray(c) for c in classes]


def leaf_prediction(model: Any, leaf_id: int, tasks: List[int]) -> Dict[int, int]:
    vals = model.tree_.value[leaf_id]
    cls = classes_by_output(model)
    preds = {}
    for out_i, task in enumerate(tasks):
        arr = np.asarray(vals[out_i])[: len(cls[out_i])]
        preds[task] = int(cls[out_i][int(np.argmax(arr))])
    return preds


def branch_constraint(threshold: float, branch: str):
    if branch == "left":
        allowed = [b for b in (0, 1) if b <= threshold]
    elif branch == "right":
        allowed = [b for b in (0, 1) if b > threshold]
    else:
        raise ValueError(branch)
    if allowed == [0]:
        return 0
    if allowed == [1]:
        return 1
    if allowed == [0, 1]:
        return None
    return "impossible"


def export_tree_rules(model: Any, tasks: List[int], suffix: str) -> List[Dict[str, Any]]:
    tree = model.tree_
    rules = []

    def rec(node_id: int, constraints: Dict[int, int], path: List[Dict[str, Any]]):
        left, right = int(tree.children_left[node_id]), int(tree.children_right[node_id])
        if left == right or left < 0:
            value_bits = ["0"] * 30
            mask_bits = ["0"] * 30
            for local_idx, bit_val in sorted(constraints.items()):
                value_bits[local_idx] = "1" if bit_val else "0"
                mask_bits[local_idx] = "1"
            key_bits = "".join(value_bits)
            mask = "".join(mask_bits)
            pred = leaf_prediction(model, node_id, tasks)
            rules.append({
                "table": f"tb_hwimo_tree_{suffix}",
                "action": f"set_pred_{suffix}",
                "leaf_id": int(node_id),
                "key_bits": key_bits,
                "mask_bits": mask,
                "key_hex": bitstr_to_hex(key_bits),
                "mask_hex": bitstr_to_hex(mask),
                "priority": 1,
                "pred": {f"task{t}": int(pred[t]) for t in tasks},
                "params": [int(pred[t]) for t in tasks],
                "path": path,
            })
            return

        feat, thr = int(tree.feature[node_id]), float(tree.threshold[node_id])
        for branch, child in (("left", left), ("right", right)):
            c = branch_constraint(thr, branch)
            if c == "impossible":
                continue
            nc = dict(constraints)
            npth = list(path)
            npth.append({"node": int(node_id), "local_feature_idx": feat, "threshold": thr, "branch": branch, "constraint": None if c is None else int(c)})
            if c is not None:
                old = nc.get(feat)
                if old is not None and old != c:
                    continue
                nc[feat] = int(c)
            rec(int(child), nc, npth)

    rec(0, {}, [])
    return rules


def extract_X(lines: List[str], compact_indices: List[int]) -> np.ndarray:
    return np.asarray([[1 if s[i] == "1" else 0 for i in compact_indices] for s in lines], dtype=np.uint8)


def rule_predict_one(key_bits: str, rules: List[Dict[str, Any]], tasks: List[int]) -> Dict[int, int]:
    key = int(key_bits, 2)
    for r in rules:
        mask = int(r["mask_bits"], 2)
        val = int(r["key_bits"], 2)
        if (key & mask) == (val & mask):
            return {t: int(v) for t, v in zip(tasks, r["params"])}
    return {t: 0 for t in tasks}


@dataclass
class ClusterArtifacts:
    cluster: str
    suffix: str
    tasks: List[int]
    model: Any
    selected_global_indices: List[int]
    selected_compact_indices: List[int]
    p4_bit_indices: List[int]
    rules: List[Dict[str, Any]]


def load_cluster(root: Path, cfg: Dict[str, Any]) -> ClusterArtifacts:
    d = root / "models" / "top_030" / cfg["cluster"]
    selected_global = np.load(d / "selected_feature_indices.npy").astype(int).tolist()
    selected_compact = [global_to_compact(g) for g in selected_global]
    p4_bits = [compact_to_p4_bit(c) for c in selected_compact]
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        with open(d / "model_mtl_feature_selected_top_30.pkl", "rb") as f:
            model = pickle.load(f)
    if int(getattr(model, "n_features_in_", -1)) != 30:
        raise RuntimeError(f"{cfg['cluster']}: expected n_features_in_=30, got {getattr(model, 'n_features_in_', None)}")
    if int(getattr(model, "n_outputs_", 1)) != len(cfg["tasks"]):
        raise RuntimeError(f"{cfg['cluster']}: expected n_outputs_={len(cfg['tasks'])}, got {getattr(model, 'n_outputs_', None)}")
    rules = export_tree_rules(model, list(cfg["tasks"]), cfg["suffix"])
    return ClusterArtifacts(cfg["cluster"], cfg["suffix"], list(cfg["tasks"]), model, selected_global, selected_compact, p4_bits, rules)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_exports(root: Path, program: str, clusters: List[ClusterArtifacts]):
    out_dir = root / "exports" / program
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "program": program,
        "feature_mode": "top_030",
        "packet_feature_width": 153,
        "model_input_width": 30,
        "selected_indices_are_global_169_bit_indices": True,
        "removed_global_indices": {"ip_id": list(range(126, 142))},
        "global_to_compact_mapping": "0..125 -> same; 142..168 -> index-16",
        "clusters": [],
        "total_entries": 0,
    }
    for c in clusters:
        table = f"tb_hwimo_tree_{c.suffix}"
        table_json = {
            "table": table,
            "action": f"set_pred_{c.suffix}",
            "cluster": c.cluster,
            "tasks": c.tasks,
            "selected_global_indices": c.selected_global_indices,
            "selected_compact_153_indices": c.selected_compact_indices,
            "p4_header_bit_indices": c.p4_bit_indices,
            "entries": c.rules,
        }
        write_json(out_dir / f"{table}.json", table_json)
        manifest["clusters"].append({
            "cluster": c.cluster,
            "tasks": c.tasks,
            "suffix": c.suffix,
            "table": table,
            "action": f"set_pred_{c.suffix}",
            "num_entries": len(c.rules),
            "selected_global_indices": c.selected_global_indices,
            "selected_compact_153_indices": c.selected_compact_indices,
            "p4_header_bit_indices": c.p4_bit_indices,
            "tree_nodes": int(c.model.tree_.node_count),
            "tree_depth": int(c.model.tree_.max_depth),
        })
        manifest["total_entries"] += len(c.rules)
    write_json(out_dir / "manifest.json", manifest)


def write_commands(root: Path, program: str, clusters: List[ClusterArtifacts]) -> Path:
    out = root / "commands" / f"{program}_commands.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# BMv2 simple_switch_CLI commands for {program}\n\n")
        for c in clusters:
            table = f"MyIngress.tb_hwimo_tree_{c.suffix}"
            action = f"MyIngress.set_pred_{c.suffix}"
            f.write(f"# {c.cluster}: tasks {c.tasks}, entries {len(c.rules)}\n")
            for r in c.rules:
                match = f"{r['key_hex']}&&&{r['mask_hex']}"
                params = " ".join(str(x) for x in r["params"])
                f.write(f"table_add {table} {action} {match} => {params} {int(r['priority'])}\n")
            f.write("\n")
    return out


def p4_type_for_task(t: int) -> str:
    return f"bit<{TASK_WIDTH[t]}>"


def generate_p4(program: str, clusters: List[ClusterArtifacts]) -> str:
    meta_fields = "\n".join([f"    bit<30> key_{c.suffix};" for c in clusters])
    actions, tables, projection_and_apply = [], [], []
    for c in clusters:
        params = ", ".join([f"{p4_type_for_task(t)} v{t}" for t in c.tasks])
        assigns = "\n".join([f"        hdr.prediction.pred_task{t} = v{t};" for t in c.tasks])
        actions.append(f"""
    action set_pred_{c.suffix}({params}) {{
{assigns}
    }}
""")
        tables.append(f"""
    table tb_hwimo_tree_{c.suffix} {{
        key = {{
            meta.key_{c.suffix}: ternary;
        }}
        actions = {{
            set_pred_{c.suffix};
            NoAction;
        }}
        size = {max(1, len(c.rules))};
        default_action = NoAction();
    }}
""")
        proj = [f"        meta.key_{c.suffix} = 0;"]
        for local_i, p4bit in enumerate(c.p4_bit_indices):
            dst_bit = 29 - local_i
            proj.append(f"        meta.key_{c.suffix}[{dst_bit}:{dst_bit}] = hdr.features.bits[{p4bit}:{p4bit}];")
        proj.append(f"        tb_hwimo_tree_{c.suffix}.apply();")
        projection_and_apply.append("\n".join(proj))

    return f'''#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x0800;
const bit<8> IP_PROTO_HWIMO = 200;

header ethernet_t {{
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}}

header ipv4_t {{
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}}

header hwimo_t {{
    bit<32> sample_id;
}}

header features_t {{
    bit<153> bits;
    bit<7>   pad;
}}

header prediction_t {{
    bit<1> pred_task1;
    bit<1> pred_task2;
    bit<3> pred_task3;
    bit<3> pred_task4;
    bit<2> pred_task5;
    bit<2> pred_task6;
    bit<2> pred_task7;
    bit<2> pad;
}}

header tcp_t {{
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}}

struct headers_t {{
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    hwimo_t      hwimo;
    features_t   features;
    prediction_t prediction;
    tcp_t        tcp;
}}

struct metadata_t {{
{meta_fields}
}}

parser MyParser(packet_in packet, out headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) {{
    state start {{ transition parse_ethernet; }}
    state parse_ethernet {{
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {{ TYPE_IPV4: parse_ipv4; default: accept; }}
    }}
    state parse_ipv4 {{
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {{ IP_PROTO_HWIMO: parse_hwimo; default: accept; }}
    }}
    state parse_hwimo {{ packet.extract(hdr.hwimo); transition parse_features; }}
    state parse_features {{ packet.extract(hdr.features); transition parse_prediction; }}
    state parse_prediction {{ packet.extract(hdr.prediction); transition parse_tcp; }}
    state parse_tcp {{ packet.extract(hdr.tcp); transition accept; }}
}}

control MyVerifyChecksum(inout headers_t hdr, inout metadata_t meta) {{ apply {{ }} }}

control MyIngress(inout headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) {{
{''.join(actions)}
{''.join(tables)}
    apply {{
        if (hdr.features.isValid()) {{
{chr(10).join(projection_and_apply)}
        }}
        if (standard_metadata.ingress_port == 0) {{
            standard_metadata.egress_spec = 1;
        }} else {{
            standard_metadata.egress_spec = 0;
        }}
    }}
}}

control MyEgress(inout headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) {{ apply {{ }} }}
control MyComputeChecksum(inout headers_t hdr, inout metadata_t meta) {{ apply {{ }} }}

control MyDeparser(packet_out packet, in headers_t hdr) {{
    apply {{
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.hwimo);
        packet.emit(hdr.features);
        packet.emit(hdr.prediction);
        packet.emit(hdr.tcp);
    }}
}}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(), MyComputeChecksum(), MyDeparser()) main;
'''


def write_p4(root: Path, program: str, clusters: List[ClusterArtifacts]) -> Path:
    filename = {
        "Hwimo_top030_proposed": "hwimo_top030_proposed_bmv2.p4",
        "Hwimo_top030_mtl_all": "hwimo_top030_mtl_all_bmv2.p4",
    }[program]
    out = root / "p4" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_p4(program, clusters), encoding="utf-8")
    return out


def write_expected(root: Path, program: str, clusters: List[ClusterArtifacts], test_lines: List[str], labels: Dict[int, List[int]]):
    n = len(test_lines)
    pred_by_task = {t: [0] * n for t in range(1, 8)}
    for c in clusters:
        X = extract_X(test_lines, c.selected_compact_indices)
        y = np.asarray(c.model.predict(X))
        if len(c.tasks) == 1:
            y = y.reshape(-1, 1)
        for out_i, task in enumerate(c.tasks):
            pred_by_task[task] = [int(v) for v in y[:, out_i].tolist()]

    out = root / "expected" / f"expected_{program}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id"] + [f"pred_task{t}" for t in range(1, 8)]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            row = {"sample_id": i}
            for t in range(1, 8):
                row[f"pred_task{t}"] = pred_by_task[t][i]
            w.writerow(row)

    accs = {}
    for t in range(1, 8):
        correct = sum(1 for a, b in zip(pred_by_task[t], labels[t]) if int(a) == int(b))
        accs[f"task{t}"] = {"correct": correct, "total": n, "accuracy": correct / n if n else 0.0}
    summary = {"program": program, "n_test": n, "accuracy": accs, "macro_accuracy": sum(v["accuracy"] for v in accs.values()) / 7.0}
    write_json(root / "expected" / f"expected_{program}_summary.json", summary)
    return out, summary


def self_check_rules(program: str, clusters: List[ClusterArtifacts], test_lines: List[str]):
    print(f"=== rule self-check: {program} ===")
    for c in clusters:
        X = extract_X(test_lines, c.selected_compact_indices)
        model_pred = np.asarray(c.model.predict(X))
        if len(c.tasks) == 1:
            model_pred = model_pred.reshape(-1, 1)
        mismatches, first = 0, None
        for i, row in enumerate(X):
            key_bits = "".join("1" if int(b) else "0" for b in row.tolist())
            rp = rule_predict_one(key_bits, c.rules, c.tasks)
            for out_i, task in enumerate(c.tasks):
                if int(model_pred[i, out_i]) != int(rp[task]):
                    mismatches += 1
                    if first is None:
                        first = {"sample_id": i, "task": task, "model": int(model_pred[i, out_i]), "rule": int(rp[task]), "key_bits": key_bits}
                    break
        print(f"{c.cluster:24s} entries={len(c.rules):5d} mismatches={mismatches}")
        if first is not None:
            print("  first mismatch:", first)
            raise RuntimeError(f"rule self-check failed for {program}/{c.cluster}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/mncgpu4/COMPACT/hwimo_bmv2_top030"))
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    root = args.root
    data_dir = root / "data"
    input_path = args.input or ((data_dir / "test_input_153.txt") if (data_dir / "test_input_153.txt").exists() else (data_dir / "153_input.txt"))

    meta = read_meta(data_dir / "test_meta.txt")
    start_idx = int(meta.get("start_idx", "0"))
    n_test = int(meta["n_test"])
    all_lines = read_feature_lines(input_path)
    if len(all_lines) < start_idx + n_test:
        raise RuntimeError(f"{input_path} rows={len(all_lines)}, need {start_idx+n_test}")
    test_lines = all_lines[start_idx:start_idx+n_test]
    labels = read_labels(root, start_idx, n_test)

    print("root:", root)
    print("input:", input_path)
    print("all rows:", len(all_lines))
    print("start_idx:", start_idx)
    print("n_test:", n_test)

    all_outputs = {}
    for program, cfgs in PROGRAMS.items():
        print("=" * 100)
        print("PROGRAM:", program)
        clusters = [load_cluster(root, cfg) for cfg in cfgs]
        for c in clusters:
            print(f"{c.cluster:24s} suffix={c.suffix:15s} tasks={c.tasks} entries={len(c.rules)} nodes={c.model.tree_.node_count} depth={c.model.tree_.max_depth}")
            print("  selected_global:", c.selected_global_indices)
            print("  selected_compact:", c.selected_compact_indices)
            print("  p4_bits:", c.p4_bit_indices)

        write_exports(root, program, clusters)
        cmd_path = write_commands(root, program, clusters)
        p4_path = write_p4(root, program, clusters)
        exp_path, summary = write_expected(root, program, clusters, test_lines, labels)
        if not args.skip_self_check:
            self_check_rules(program, clusters, test_lines)

        all_outputs[program] = {
            "commands": str(cmd_path),
            "p4": str(p4_path),
            "expected": str(exp_path),
            "macro_accuracy": summary["macro_accuracy"],
            "total_entries": sum(len(c.rules) for c in clusters),
        }
        print("commands:", cmd_path)
        print("p4:", p4_path)
        print("expected:", exp_path)
        print("macro accuracy:", f"{summary['macro_accuracy']*100:.4f}%")

    write_json(root / "results" / "generate_top030_bmv2_artifacts_summary.json", all_outputs)

    print("=" * 100)
    print("[DONE]")
    for program, o in all_outputs.items():
        print(program)
        print("  entries:", o["total_entries"])
        print("  macro accuracy:", f"{o['macro_accuracy']*100:.4f}%")
        print("  p4:", o["p4"])
        print("  commands:", o["commands"])
        print("  expected:", o["expected"])


if __name__ == "__main__":
    main()
