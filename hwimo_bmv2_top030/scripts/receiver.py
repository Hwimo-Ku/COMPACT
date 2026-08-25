#!/usr/bin/env python3
"""
receiver.py

HWIMO BMv2 receiver with old validator-compatible accuracy.

This receiver:
  - writes MALOI-style RecvPkt_*.txt
  - compares P4 prediction with expected_Hwimo_*.csv
  - computes full_match_rate exactly like validate_top030_bmv2_packet_replay.py
  - computes P4-vs-label task-wise accuracy on received packets
  - has no idle/max timeout; exits only after expected_count packets are received
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scapy.all import IP, sniff
except ModuleNotFoundError:
    raise SystemExit(
        "[ERROR] scapy is not installed.\n"
        "Install it with:\n"
        "  python3 -m pip install --user scapy\n"
    )


TASKS = list(range(1, 8))

PROGRAMS = {
    "proposed": {
        "name": "Hwimo_top030_proposed",
        "expected": "expected_Hwimo_top030_proposed.csv",
        "recv_log": "RecvPkt_Hwimo_top030_proposed.txt",
        "summary_json": "bmv2_top030_proposed_receiver_summary.json",
        "pred_csv": "bmv2_top030_proposed_receiver_predictions.csv",
    },
    "low030_mtl_all": {
        "name": "Hwimo_low030_mtl_all",
        "expected": "expected_Hwimo_low030_mtl_all.csv",
        "recv_log": "RecvPkt_Hwimo_low030_mtl_all.txt",
        "summary_json": "bmv2_low030_mtl_all_receiver_summary.json",
        "pred_csv": "bmv2_low030_mtl_all_receiver_predictions.csv",
    },
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


def read_expected(path: Path) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["sample_id"])
            out[sid] = {
                t: int(row[f"pred_task{t}"])
                for t in TASKS
            }
    return out


def read_labels(root: Path, start_idx: int, n_test: int) -> Dict[int, List[int]]:
    labels: Dict[int, List[int]] = {}

    for t in TASKS:
        p = root / "data" / f"task{t}_label_i.txt"
        vals: List[int] = []

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s:
                    vals.append(int(s.split()[0]))

        if len(vals) < start_idx + n_test:
            raise RuntimeError(f"{p} rows={len(vals)}, need {start_idx + n_test}")

        labels[t] = vals[start_idx:start_idx + n_test]

    return labels


def parse_prediction_word(v: int) -> Dict[int, int]:
    return {
        1: (v >> 15) & 0x1,
        2: (v >> 14) & 0x1,
        3: (v >> 11) & 0x7,
        4: (v >> 8) & 0x7,
        5: (v >> 6) & 0x3,
        6: (v >> 4) & 0x3,
        7: (v >> 2) & 0x3,
    }


def extract_hwimo_packet(pkt) -> Optional[Tuple[int, Dict[int, int], int]]:
    if IP not in pkt:
        return None

    ip = pkt[IP]

    if int(ip.proto) != 200:
        return None

    payload = bytes(ip.payload)

    if len(payload) < 26:
        return None

    sample_id = int.from_bytes(payload[0:4], byteorder="big", signed=False)
    pred_word = int.from_bytes(payload[24:26], byteorder="big", signed=False)
    pred = parse_prediction_word(pred_word)

    return sample_id, pred, pred_word


def pred_tuple_str(pred: Dict[int, int]) -> str:
    return "(" + " ".join(str(int(pred[t])) for t in TASKS) + ")"


def expected_match(pred: Dict[int, int], exp: Dict[int, int]) -> bool:
    return all(int(pred[t]) == int(exp[t]) for t in TASKS)


def write_header(fp) -> None:
    fp.write(f"{'No':<10s} {'Predictions':<22s} {'Accuracy [%]':>12s}\n")
    fp.flush()


def write_packet_row(fp, sample_id: int, pred: Dict[int, int], running_acc: float) -> None:
    fp.write(f"{sample_id:<10d} {pred_tuple_str(pred):<22s} {running_acc:>12.3f}\n")
    fp.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--program", choices=sorted(PROGRAMS.keys()), required=True)
    ap.add_argument("--iface-out", default="veth3")
    ap.add_argument("--limit", type=int, default=0, help="0 means all test packets")
    ap.add_argument("--start", type=int, default=0, help="relative sample_id start within test split")
    args = ap.parse_args()

    root = args.root.resolve()
    data_dir = root / "data"
    expected_dir = root / "expected"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prog = PROGRAMS[args.program]

    meta = read_meta(data_dir / "test_meta.txt")
    start_idx = int(meta.get("start_idx", "0"))
    n_test = int(meta["n_test"])

    total = n_test if args.limit == 0 else min(args.limit, n_test - args.start)
    sample_ids = list(range(args.start, args.start + total))
    expected_count = len(sample_ids)

    if expected_count <= 0:
        raise RuntimeError("no packets selected")

    expected = read_expected(expected_dir / prog["expected"])
    labels = read_labels(root, start_idx=start_idx, n_test=n_test)

    recv_log_path = results_dir / prog["recv_log"]
    pred_csv_path = results_dir / prog["pred_csv"]
    summary_json_path = results_dir / prog["summary_json"]

    received: Dict[int, Dict[int, int]] = {}
    pred_words: Dict[int, int] = {}
    bad_parse = 0

    rows = []
    first_bad_prediction = []

    running_received = 0
    running_task_correct = {t: 0 for t in TASKS}

    print("=" * 80)
    print(f"Receiver validation: {prog['name']}")
    print("=" * 80)
    print(f"iface out:       {args.iface_out}")
    print(f"expected:        {expected_dir / prog['expected']}")
    print("=" * 80)

    with open(recv_log_path, "w", encoding="utf-8") as fp:
        write_header(fp)

        while True:
            if len(received) >= expected_count:
                break

            pkts = sniff(
                iface=args.iface_out,
                count=1,
                timeout=1,
                store=True,
            )

            if not pkts:
                continue

            try:
                parsed = extract_hwimo_packet(pkts[0])
            except Exception:
                bad_parse += 1
                continue

            if parsed is None:
                continue

            sid, pred, pred_word = parsed

            if sid not in sample_ids:
                continue

            if sid in received:
                continue

            if sid not in expected:
                continue

            exp = expected[sid]
            match = expected_match(pred, exp)

            received[sid] = pred
            pred_words[sid] = pred_word

            running_received += 1
            for t in TASKS:
                if int(pred[t]) == int(labels[t][sid]):
                    running_task_correct[t] += 1

            running_macro_acc = (
                sum(running_task_correct[t] / running_received for t in TASKS)
                / len(TASKS)
                * 100.0
            )

            write_packet_row(fp, sid, pred, running_macro_acc)

        missing_all = [sid for sid in sample_ids if sid not in received]

        bad_prediction = 0
        p4_label_correct = {t: 0 for t in TASKS}

        for sid in sample_ids:
            exp = expected[sid]
            pred = received.get(sid)

            row = {
                "sample_id": sid,
                "received": int(pred is not None),
            }

            if pred is None:
                row["match_expected"] = 0
                row["pred_word"] = ""

                for t in TASKS:
                    row[f"expected_task{t}"] = exp[t]
                    row[f"p4_task{t}"] = ""
                    row[f"label_task{t}"] = labels[t][sid]

                rows.append(row)
                continue

            ok = expected_match(pred, exp)

            row["match_expected"] = int(ok)
            row["pred_word"] = int(pred_words[sid])

            if not ok:
                bad_prediction += 1
                if len(first_bad_prediction) < 10:
                    first_bad_prediction.append(
                        {
                            "sample_id": sid,
                            "expected": {f"task{t}": exp[t] for t in TASKS},
                            "p4": {f"task{t}": pred[t] for t in TASKS},
                        }
                    )

            for t in TASKS:
                row[f"expected_task{t}"] = exp[t]
                row[f"p4_task{t}"] = pred[t]
                row[f"label_task{t}"] = labels[t][sid]

                if int(pred[t]) == int(labels[t][sid]):
                    p4_label_correct[t] += 1

            rows.append(row)

        received_count = len([sid for sid in sample_ids if sid in received])
        full_matches = expected_count - len(missing_all) - bad_prediction
        full_match_rate = full_matches / expected_count if expected_count else 0.0

        label_acc = {
            f"task{t}": {
                "correct": int(p4_label_correct[t]),
                "total": int(received_count),
                "accuracy": (p4_label_correct[t] / received_count) if received_count else 0.0,
            }
            for t in TASKS
        }

        macro_acc = (
            sum(v["accuracy"] for v in label_acc.values()) / 7.0
            if received_count
            else 0.0
        )

        fp.write("\n=== Expected Validation ===\n")
        fp.write(f"Selected Packets: {expected_count}\n")
        fp.write(f"Received Packets: {received_count}/{expected_count}\n")
        fp.write(f"Missing Packets: {len(missing_all)}\n")
        fp.write(f"Bad Parse: {bad_parse}\n")
        fp.write(f"Bad Prediction: {bad_prediction}\n")
        fp.write(f"Full Matches: {full_matches}/{expected_count}\n")
        fp.write(f"Full Match Rate: {full_match_rate * 100.0:.4f}%\n")

        fp.write("\n=== P4-vs-label accuracy on received packets ===\n")
        for t in TASKS:
            a = label_acc[f"task{t}"]
            fp.write(
                f"Task {t}: {a['accuracy'] * 100.0:.4f}% "
                f"({a['correct']}/{a['total']})\n"
            )
        fp.write(f"Macro Accuracy: {macro_acc * 100.0:.4f}%\n")
        fp.flush()

    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = (
            ["sample_id", "received", "match_expected", "pred_word"]
            + [f"expected_task{t}" for t in TASKS]
            + [f"p4_task{t}" for t in TASKS]
            + [f"label_task{t}" for t in TASKS]
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "program": prog["name"],
        "selected_packets": expected_count,
        "received": received_count,
        "missing": len(missing_all),
        "bad_parse": bad_parse,
        "bad_prediction": bad_prediction,
        "full_matches": full_matches,
        "full_match_rate": full_match_rate,
        "csv": str(pred_csv_path),
        "recv_log": str(recv_log_path),
        "first_missing": missing_all[:20],
        "first_bad_prediction": first_bad_prediction,
        "p4_vs_label_accuracy_on_received": label_acc,
        "p4_vs_label_macro_accuracy_on_received": macro_acc,
    }

    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"selected:        {expected_count}")
    print(f"received:        {received_count}")
    print(f"missing:         {len(missing_all)}")
    print(f"bad parse:       {bad_parse}")
    print(f"bad prediction:  {bad_prediction}")
    print(f"full matches:    {full_matches} / {expected_count} = {full_match_rate * 100.0:.4f}%")
    print(f"recv log:        {recv_log_path}")
    print(f"csv:             {pred_csv_path}")
    print(f"summary:         {summary_json_path}")
    print("P4-vs-label accuracy on received packets:")
    for t in TASKS:
        a = label_acc[f"task{t}"]
        print(f"  task{t}: {a['accuracy'] * 100.0:.4f}% ({a['correct']}/{a['total']})")
    print(f"  macro: {macro_acc * 100.0:.4f}%")

    if missing_all:
        print("first missing:", missing_all[:20])

    if first_bad_prediction:
        print("first bad prediction:", first_bad_prediction[:3])


if __name__ == "__main__":
    main()
