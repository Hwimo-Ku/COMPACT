#!/usr/bin/env python3
"""
recv_progress.py

Reads the latest RecvPkt_*.txt data row and prints compact RX progress.

Works with either format:
  No Received SampleID Predictions Accuracy
  No SampleID Predictions Accuracy
  No Predictions Accuracy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


PROGRAMS = {
    "proposed": "RecvPkt_Hwimo_top030_proposed.txt",
    "low030_mtl_all": "RecvPkt_Hwimo_low030_mtl_all.txt",
}


def default_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return here.parent


def read_meta(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}

    if not path.exists():
        return out

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and "=" in s:
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip()

    return out


def is_data_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # data row starts with integer and contains prediction tuple
    return bool(re.match(r"^\d+\s+", s)) and "(" in s and ")" in s


def parse_last_accuracy(line: str) -> Optional[float]:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", line.strip())
    if not m:
        return None
    return float(m.group(1))


def parse_received_token(line: str) -> Optional[tuple[int, int]]:
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--program", choices=sorted(PROGRAMS.keys()), required=True)
    ap.add_argument("--expected", type=int, default=0)

    args = ap.parse_args()

    root = args.root.resolve()
    recv_path = root / "results" / PROGRAMS[args.program]

    if not recv_path.exists():
        print(f"RX progress: no RecvPkt file yet: {recv_path}")
        return

    lines = recv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines: List[str] = [line for line in lines if is_data_line(line)]

    meta = read_meta(root / "data" / "test_meta.txt")
    expected_from_meta = int(meta.get("n_test", "0")) if meta.get("n_test") else 0
    expected = args.expected or expected_from_meta

    if not data_lines:
        print(f"RX progress: 0/{expected} (0.00%)  accuracy=N/A")
        return

    last = data_lines[-1]
    acc = parse_last_accuracy(last)

    recv_token = parse_received_token(last)
    if recv_token is not None:
        received, expected_in_file = recv_token
        if expected <= 0:
            expected = expected_in_file
    else:
        received = len(data_lines)

    pct = 100.0 * received / expected if expected else 0.0
    acc_s = f"{acc:.2f}%" if acc is not None else "N/A"

    print(
        f"accuracy={acc_s}"
    )


if __name__ == "__main__":
    main()
