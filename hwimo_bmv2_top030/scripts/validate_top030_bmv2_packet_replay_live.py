#!/usr/bin/env python3
"""
BMv2 packet replay validator with optional live per-packet output for HWIMO top030 proposed and low030 MTL-all.

Expected packet path:
  sender iface -> BMv2 port 0 -> BMv2 port 1 -> receiver iface

Recommended BMv2 setup:
  simple_switch -i 0@veth0 -i 1@veth2 ...
  validator --iface-in veth1 --iface-out veth3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from scapy.all import Ether, IP, Raw, AsyncSniffer, sendp, conf
except Exception as e:
    raise SystemExit(f"[ERROR] scapy import failed: {e}\nInstall with: python3 -m pip install --user scapy")

PROGRAMS = {
    "proposed": {
        "name": "Hwimo_top030_proposed",
        "expected": "expected_Hwimo_top030_proposed.csv",
        "out_prefix": "bmv2_top030_proposed",
    },
    "low030_mtl_all": {
        "name": "Hwimo_low030_mtl_all",
        "expected": "expected_Hwimo_low030_mtl_all.csv",
        "out_prefix": "bmv2_low030_mtl_all",
    },
}

TASKS = list(range(1, 8))
TASK_NAMES = {
    1: "vpn",
    2: "tor",
    3: "service",
    4: "app",
    5: "flow_size",
    6: "flow_duration",
    7: "avg_pkt_len",
}


def read_meta(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
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
        raise RuntimeError(f"{path} expected 153-bit rows, observed widths={sorted(widths)}")
    return lines


def read_expected(path: Path) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["sample_id"])
            out[sid] = {t: int(row[f"pred_task{t}"]) for t in TASKS}
    return out


def read_labels(root: Path, start_idx: int, n_test: int) -> Dict[int, List[int]]:
    labels: Dict[int, List[int]] = {}
    for t in TASKS:
        p = root / "data" / f"task{t}_label_i.txt"
        vals: List[int] = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    vals.append(int(line.split()[0]))
        if len(vals) < start_idx + n_test:
            raise RuntimeError(f"{p} rows={len(vals)}, need {start_idx+n_test}")
        labels[t] = vals[start_idx:start_idx+n_test]
    return labels


def feature_bytes(feature_bits_153: str) -> bytes:
    if len(feature_bits_153) != 153:
        raise ValueError(f"feature length must be 153, got {len(feature_bits_153)}")
    # P4 header features_t is bit<153> bits followed by bit<7> pad.
    return int(feature_bits_153 + "0" * 7, 2).to_bytes(20, "big")


def build_packet(sample_id: int, feature_bits_153: str) -> Ether:
    payload = (
        int(sample_id).to_bytes(4, "big") +
        feature_bytes(feature_bits_153) +
        b"\x00\x00" +                  # prediction_t initialized to zero
        b"\x12\x34\x00\x50" +          # tcp src/dst ports
        b"\x00\x00\x00\x00" +          # seq
        b"\x00\x00\x00\x00" +          # ack
        b"\x50\x00" +                  # dataOffset/res/ecn/ctrl
        b"\x20\x00" +                  # window
        b"\x00\x00" +                  # checksum
        b"\x00\x00"                    # urgent
    )
    return (
        Ether(dst="02:00:00:00:00:02", src="02:00:00:00:00:01", type=0x0800) /
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=64, proto=200) /
        Raw(payload)
    )


def parse_prediction(raw: bytes) -> Tuple[int, Dict[int, int]]:
    if len(raw) < 14 + 20:
        raise ValueError("short packet")
    if int.from_bytes(raw[12:14], "big") != 0x0800:
        raise ValueError("not IPv4")
    ip0 = 14
    ihl = (raw[ip0] & 0x0F) * 4
    if len(raw) < ip0 + ihl + 4 + 20 + 2:
        raise ValueError("short HWIMO packet")
    if raw[ip0 + 9] != 200:
        raise ValueError("not HWIMO proto 200")

    off = ip0 + ihl
    sample_id = int.from_bytes(raw[off:off + 4], "big")
    pred_off = off + 4 + 20
    v = int.from_bytes(raw[pred_off:pred_off + 2], "big")

    pred = {
        1: (v >> 15) & 0x1,
        2: (v >> 14) & 0x1,
        3: (v >> 11) & 0x7,
        4: (v >> 8) & 0x7,
        5: (v >> 6) & 0x3,
        6: (v >> 4) & 0x3,
        7: (v >> 2) & 0x3,
    }
    return sample_id, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/mncgpu4/COMPACT/hwimo_bmv2_top030"))
    ap.add_argument("--program", choices=sorted(PROGRAMS), required=True)
    ap.add_argument("--iface-in", default="veth1")
    ap.add_argument("--iface-out", default="veth3")
    ap.add_argument("--limit", type=int, default=0, help="0 means all test packets")
    ap.add_argument("--start", type=int, default=0, help="relative sample_id start within the test split")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--send-interval", type=float, default=0.002)
    ap.add_argument("--retry-send-interval", type=float, default=0.004)
    ap.add_argument("--batch-timeout", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--post-wait", type=float, default=1.0)
    ap.add_argument("--live-print", choices=["none", "all", "mismatch"], default="none",
                    help="Print per-packet prediction as packets are received. all=print every selected packet; mismatch=print only P4-vs-expected mismatches.")
    ap.add_argument("--live-max", type=int, default=50,
                    help="Maximum number of live per-packet lines to print. 0 means unlimited.")
    ap.add_argument("--live-every", type=int, default=1,
                    help="When live-print=all, print one packet every N received selected packets.")
    args = ap.parse_args()

    root = args.root
    prog = PROGRAMS[args.program]
    data_dir = root / "data"
    input_path = data_dir / "test_input_153.txt"
    if not input_path.exists():
        input_path = data_dir / "153_input.txt"
    expected_path = root / "expected" / prog["expected"]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta(data_dir / "test_meta.txt")
    start_idx = int(meta.get("start_idx", "0"))
    n_test = int(meta["n_test"])
    all_lines = read_feature_lines(input_path)
    if len(all_lines) < start_idx + n_test:
        raise RuntimeError(f"{input_path} rows={len(all_lines)}, need {start_idx+n_test}")
    test_lines = all_lines[start_idx:start_idx + n_test]
    expected = read_expected(expected_path)
    labels = read_labels(root, start_idx, n_test)

    total = n_test if args.limit == 0 else min(args.limit, n_test - args.start)
    sample_ids = list(range(args.start, args.start + total))
    if not sample_ids:
        raise RuntimeError("no packets selected")

    print("=" * 80)
    print(f"BMv2 packet replay validation: {prog['name']}")
    print("=" * 80)
    print("root:          ", root)
    print("iface in:      ", args.iface_in)
    print("iface out:     ", args.iface_out)
    print("input:         ", input_path)
    print("expected:      ", expected_path)
    print("test start_idx:", start_idx)
    print("test n:        ", n_test)
    print("selected n:    ", len(sample_ids))
    print("batch size:    ", args.batch_size)
    print("send interval: ", args.send_interval)
    print("retries:       ", args.retries)
    print("live print:    ", args.live_print)
    print("live max:      ", args.live_max)
    print("live every:    ", args.live_every)
    print("=" * 80)

    received: Dict[int, Dict[int, int]] = {}
    bad_parse = 0
    live_printed = 0
    live_seen_selected = 0
    selected_set = set(sample_ids)

    def fmt_vec(d: Dict[int, int]) -> str:
        return " ".join(f"t{t}={d[t]}" for t in TASKS)

    def cb(pkt):
        nonlocal bad_parse, live_printed, live_seen_selected
        try:
            sid, pred = parse_prediction(bytes(pkt))
        except Exception:
            bad_parse += 1
            return
        if sid in expected:
            received[sid] = pred

            if sid in selected_set:
                live_seen_selected += 1
                exp = expected[sid]
                lab = {t: labels[t][sid] for t in TASKS}
                match_expected = all(int(pred[t]) == int(exp[t]) for t in TASKS)
                match_label = {t: int(pred[t]) == int(lab[t]) for t in TASKS}

                should_print = False
                if args.live_print == "all":
                    if args.live_every <= 1 or (live_seen_selected % args.live_every == 0):
                        should_print = True
                elif args.live_print == "mismatch":
                    if not match_expected:
                        should_print = True

                if should_print and (args.live_max == 0 or live_printed < args.live_max):
                    live_printed += 1
                    label_marks = " ".join(
                        f"t{t}:{'OK' if match_label[t] else 'WRONG'}"
                        for t in TASKS
                    )
                    print(
                        f"[PKT {live_seen_selected:06d}] sample_id={sid} "
                        f"p4=[{fmt_vec(pred)}] "
                        f"expected=[{fmt_vec(exp)}] "
                        f"label=[{fmt_vec(lab)}] "
                        f"p4==expected={'YES' if match_expected else 'NO'} "
                        f"p4_vs_label=[{label_marks}]",
                        flush=True,
                    )

    # Avoid libpcap dependency when possible.
    try:
        conf.use_pcap = False
    except Exception:
        pass

    sniffer = AsyncSniffer(iface=args.iface_out, prn=cb, store=False)
    sniffer.start()
    time.sleep(0.5)

    try:
        sent_total = 0
        batches = [sample_ids[i:i + args.batch_size] for i in range(0, len(sample_ids), args.batch_size)]
        for bi, batch in enumerate(batches, 1):
            missing = [sid for sid in batch if sid not in received]
            for attempt in range(args.retries + 1):
                if not missing:
                    break
                interval = args.send_interval if attempt == 0 else args.retry_send_interval
                # print(f"batch {bi:04d}/{len(batches):04d} attempt {attempt+1}/{args.retries+1}: send={len(missing)} received_total={len(received)}/{len(sample_ids)}")
                for sid in missing:
                    pkt = build_packet(sid, test_lines[sid])
                    sendp(pkt, iface=args.iface_in, verbose=False)
                    sent_total += 1
                    if interval > 0:
                        time.sleep(interval)
                time.sleep(args.batch_timeout)
                missing = [sid for sid in batch if sid not in received]
        time.sleep(args.post_wait)
    finally:
        try:
            sniffer.stop()
        except Exception:
            pass

    missing_all = [sid for sid in sample_ids if sid not in received]

    bad_prediction = 0
    first_bad = []
    rows = []
    p4_label_correct = {t: 0 for t in TASKS}

    for sid in sample_ids:
        exp = expected[sid]
        pred = received.get(sid)
        row = {"sample_id": sid, "received": int(pred is not None)}
        if pred is None:
            row["match_expected"] = 0
            for t in TASKS:
                row[f"expected_task{t}"] = exp[t]
                row[f"p4_task{t}"] = ""
                row[f"label_task{t}"] = labels[t][sid]
            rows.append(row)
            continue

        ok = True
        for t in TASKS:
            row[f"expected_task{t}"] = exp[t]
            row[f"p4_task{t}"] = pred[t]
            row[f"label_task{t}"] = labels[t][sid]
            if int(pred[t]) != int(exp[t]):
                ok = False
            if int(pred[t]) == int(labels[t][sid]):
                p4_label_correct[t] += 1
        row["match_expected"] = int(ok)
        if not ok:
            bad_prediction += 1
            if len(first_bad) < 10:
                first_bad.append({"sample_id": sid, "expected": exp, "p4": pred})
        rows.append(row)

    csv_path = out_dir / f"{prog['out_prefix']}_packet_validation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = (
            ["sample_id", "received", "match_expected"] +
            [f"expected_task{t}" for t in TASKS] +
            [f"p4_task{t}" for t in TASKS] +
            [f"label_task{t}" for t in TASKS]
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    denom = len(sample_ids)
    received_count = len([sid for sid in sample_ids if sid in received])
    label_acc = {
        f"task{t}": {
            "correct": p4_label_correct[t],
            "total": received_count,
            "accuracy": (p4_label_correct[t] / received_count) if received_count else 0.0,
        }
        for t in TASKS
    }
    summary = {
        "program": prog["name"],
        "selected_packets": denom,
        "received": received_count,
        "missing": len(missing_all),
        "bad_parse": bad_parse,
        "bad_prediction": bad_prediction,
        "full_matches": denom - len(missing_all) - bad_prediction,
        "full_match_rate": (denom - len(missing_all) - bad_prediction) / denom if denom else 0.0,
        "csv": str(csv_path),
        "first_missing": missing_all[:20],
        "first_bad_prediction": first_bad,
        "p4_vs_label_accuracy_on_received": label_acc,
        "p4_vs_label_macro_accuracy_on_received": sum(v["accuracy"] for v in label_acc.values()) / 7.0 if received_count else 0.0,
    }
    summary_path = out_dir / f"{prog['out_prefix']}_packet_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"selected:        {denom}")
    print(f"received:        {received_count}")
    print(f"missing:         {len(missing_all)}")
    print(f"bad parse:       {bad_parse}")
    print(f"bad prediction:  {bad_prediction}")
    print(f"full matches:    {summary['full_matches']} / {denom} = {summary['full_match_rate']*100:.4f}%")
    print(f"csv:             {csv_path}")
    print(f"summary:         {summary_path}")
    print("P4-vs-label accuracy on received packets:")
    for t in TASKS:
        a = label_acc[f"task{t}"]
        print(f"  task{t}: {a['accuracy']*100:.4f}% ({a['correct']}/{a['total']})")
    print(f"  macro: {summary['p4_vs_label_macro_accuracy_on_received']*100:.4f}%")
    if missing_all:
        print("first missing:", missing_all[:20])
    if first_bad:
        print("first bad prediction:", first_bad[:3])


if __name__ == "__main__":
    main()
