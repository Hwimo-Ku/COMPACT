#!/usr/bin/env python3
"""
sender.py

HWIMO BMv2 sender.

Behavior:
  - Sends HWIMO feature packets to BMv2 input interface.
  - Does not print per-packet details.
  - Prints only TX progress percentage on one terminal line.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

try:
    from scapy.all import Ether, IP, Raw, sendp, conf
except Exception as e:
    raise SystemExit(
        f"[ERROR] scapy import failed: {e}\n"
        "Install it with:\n"
        "  python3 -m pip install --user scapy\n"
    )


PROGRAMS = {
    "proposed": "Hwimo_top030_proposed",
    "low030_mtl_all": "Hwimo_low030_mtl_all",
}


def default_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return here.parent


def read_meta(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and "=" in s:
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip()

    return out


def read_feature_lines(path: Path) -> List[str]:
    lines: List[str] = []
    widths = set()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if not s:
                continue
            widths.add(len(s))
            lines.append(s)

    if widths != {153}:
        raise RuntimeError(
            f"{path} expected 153-bit rows, observed widths={sorted(widths)}"
        )

    return lines


def feature_bytes(feature_bits_153: str) -> bytes:
    if len(feature_bits_153) != 153:
        raise ValueError(f"feature length must be 153, got {len(feature_bits_153)}")

    return int(feature_bits_153 + "0" * 7, 2).to_bytes(20, "big")


def build_packet(sample_id: int, feature_bits_153: str) -> Ether:
    payload = (
        int(sample_id).to_bytes(4, "big")
        + feature_bytes(feature_bits_153)
        + b"\x00\x00"                  # prediction_t initialized to zero
        + b"\x12\x34\x00\x50"          # tcp src/dst ports
        + b"\x00\x00\x00\x00"          # seq
        + b"\x00\x00\x00\x00"          # ack
        + b"\x50\x00"                  # dataOffset/res/ecn/ctrl
        + b"\x20\x00"                  # window
        + b"\x00\x00"                  # checksum
        + b"\x00\x00"                  # urgent
    )

    return (
        Ether(dst="02:00:00:00:00:02", src="02:00:00:00:00:01", type=0x0800)
        / IP(src="10.0.0.1", dst="10.0.0.2", ttl=64, proto=200)
        / Raw(payload)
    )


def print_progress(sent: int, total: int) -> None:
    pct = 100.0 * sent / total if total else 0.0
    print(f"\rTX progress: ({pct:6.2f}%)", end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--program", choices=sorted(PROGRAMS.keys()), required=True)
    ap.add_argument("--iface-in", default="veth1")
    ap.add_argument("--limit", type=int, default=0, help="0 means all test packets")
    ap.add_argument("--start", type=int, default=0, help="relative sample_id start within test split")
    ap.add_argument("--send-interval", type=float, default=0.01)

    # New sender progress option.
    ap.add_argument("--progress-every", type=int, default=1)

    # Old options kept for command compatibility. They are ignored.
    ap.add_argument("--print-every", type=int, default=None)
    ap.add_argument("--print-max", type=int, default=0)

    args = ap.parse_args()

    root = args.root.resolve()
    data_dir = root / "data"

    input_path = data_dir / "test_input_153.txt"
    if not input_path.exists():
        input_path = data_dir / "153_input.txt"

    meta = read_meta(data_dir / "test_meta.txt")
    start_idx = int(meta.get("start_idx", "0"))
    n_test = int(meta["n_test"])

    all_lines = read_feature_lines(input_path)

    if len(all_lines) < start_idx + n_test:
        raise RuntimeError(
            f"{input_path} rows={len(all_lines)}, need {start_idx + n_test}"
        )

    test_lines = all_lines[start_idx:start_idx + n_test]

    total = n_test if args.limit == 0 else min(args.limit, n_test - args.start)
    sample_ids = list(range(args.start, args.start + total))

    if not sample_ids:
        raise RuntimeError("no packets selected")

    try:
        conf.use_pcap = False
    except Exception:
        pass

    print("=" * 80)
    print(f"BMv2 sender: {PROGRAMS[args.program]}")
    print("=" * 80)
    print(f"root:          {root}")
    print(f"iface in:      {args.iface_in}")
    print(f"input:         {input_path}")
    print(f"send interval: {args.send_interval}")
    print("=" * 80)

    progress_every = args.progress_every
    if progress_every <= 0:
        progress_every = 0

    print_progress(0, len(sample_ids))

    sent = 0

    for sid in sample_ids:
        pkt = build_packet(sid, test_lines[sid])
        sendp(pkt, iface=args.iface_in, verbose=False)

        sent += 1

        if progress_every and (sent % progress_every == 0 or sent == len(sample_ids)):
            print_progress(sent, len(sample_ids))

        if args.send_interval > 0:
            time.sleep(args.send_interval)

    print()
    print(f"TX done")


if __name__ == "__main__":
    main()
