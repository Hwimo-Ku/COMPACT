#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/mncgpu4/COMPACT/hwimo_bmv2_top030")
TOP = ROOT / "models" / "top_030"

CLUSTERS = [
    "cluster_1",
    "cluster_2",
    "cluster_3",
    "cluster_4",
    "cluster_5_6_7",
    "cluster_1_2_3_4_5_6_7",
]

def global_to_compact(g: int) -> int:
    if 0 <= g <= 125:
        return g
    if 126 <= g <= 141:
        raise ValueError(f"global index {g} is in removed ip_id range 126..141")
    if 142 <= g <= 168:
        return g - 16
    raise ValueError(f"global index {g} outside 0..168")

out = {}

for c in CLUSTERS:
    idx_path = TOP / c / "selected_feature_indices.npy"
    idx = np.load(idx_path).astype(int).tolist()

    compact = []
    p4_bits = []

    for g in idx:
        ci = global_to_compact(g)
        compact.append(ci)
        p4_bits.append(152 - ci)

    out[c] = {
        "selected_global_indices": idx,
        "selected_compact_153_indices": compact,
        "p4_header_bit_indices": p4_bits,
    }

    print("=" * 80)
    print(c)
    print("global min/max: ", min(idx), max(idx))
    print("compact min/max:", min(compact), max(compact))
    print("global indices: ", idx)
    print("compact idx:    ", compact)
    print("p4 bit idx:     ", p4_bits)

out_path = ROOT / "models" / "top030_global_to_compact_mapping.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("=" * 80)
print("[OK] no selected feature falls in removed ip_id range 126..141")
print("wrote:", out_path)
