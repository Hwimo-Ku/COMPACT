#!/usr/bin/env bash
# run_top030_bmv2_once.sh
#
# End-to-end BMv2 run script for:
#   1) proposed        -> Hwimo_top030_proposed
#   2) low030_mtl_all -> Hwimo_low030_mtl_all
#   3) both           -> proposed then low030_mtl_all
#
# It performs:
#   - optional P4 compile
#   - clean veth setup
#   - simple_switch start
#   - simple_switch_CLI rule load
#   - packet replay validation
#   - simple_switch cleanup
#
# Default root:
#   /home/mncgpu4/COMPACT/hwimo_bmv2_top030

set -euo pipefail

ROOT="/home/mncgpu4/COMPACT/hwimo_bmv2_top030"
TARGET="both"
LIMIT=0
BATCH_SIZE=500
SEND_INTERVAL="0.002"
RETRY_SEND_INTERVAL="0.004"
BATCH_TIMEOUT="3"
RETRIES=3
THRIFT_PORT=9090
SKIP_COMPILE=0
KEEP_SWITCH=0
DEVICE_ID=30
PACKET_PRINT_EVERY=1000
PACKET_PRINT_MAX=0
PACKET_PRINT=1

usage() {
  cat <<'USAGE'
Usage:
  bash run_top030_bmv2_once.sh [TARGET] [options]

TARGET:
  proposed          Run Hwimo_top030_proposed only
  low030_mtl_all    Run Hwimo_low030_mtl_all only
  both              Run proposed then low030_mtl_all. Default.

Options:
  --root PATH                 Project root. Default: /home/mncgpu4/COMPACT/hwimo_bmv2_top030
  --limit N                   Number of test packets. 0 means all. Default: 0
  --batch-size N              Replay batch size. Default: 500
  --send-interval SEC         First-send interval. Default: 0.002
  --retry-send-interval SEC   Retry-send interval. Default: 0.004
  --batch-timeout SEC         Wait after each batch attempt. Default: 3
  --retries N                 Retry count per batch. Default: 3
  --thrift-port PORT          simple_switch thrift port. Default: 9090
  --skip-compile              Do not run p4c-bm2-ss
  --keep-switch               Do not kill simple_switch after validation
  --device-id N               BMv2 device id. Default: 30
  --packet-print-every N       Print one packet prediction every N received packets. Default: 1000
  --packet-print-max N         Max packet prediction lines. 0 means unlimited. Default: 0
  --no-packet-print            Disable per-packet prediction printing
  -h, --help                  Show this help

Examples:
  bash run_top030_bmv2_once.sh proposed

  bash run_top030_bmv2_once.sh low030_mtl_all --limit 1000

  bash run_top030_bmv2_once.sh both --limit 0 --batch-size 500 --send-interval 0.002
USAGE
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    proposed|low030_mtl_all|both)
      TARGET="$1"
      shift
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --send-interval)
      SEND_INTERVAL="$2"; shift 2 ;;
    --retry-send-interval)
      RETRY_SEND_INTERVAL="$2"; shift 2 ;;
    --batch-timeout)
      BATCH_TIMEOUT="$2"; shift 2 ;;
    --retries)
      RETRIES="$2"; shift 2 ;;
    --thrift-port)
      THRIFT_PORT="$2"; shift 2 ;;
    --skip-compile)
      SKIP_COMPILE=1; shift ;;
    --keep-switch)
      KEEP_SWITCH=1; shift ;;
    --device-id)
      DEVICE_ID="$2"; shift 2 ;;
    --packet-print-every)
      PACKET_PRINT_EVERY="$2"; shift 2 ;;
    --packet-print-max)
      PACKET_PRINT_MAX="$2"; shift 2 ;;
    --no-packet-print)
      PACKET_PRINT=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

mkdir -p "$ROOT/build" "$ROOT/logs" "$ROOT/results"

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing file: $p" >&2
    exit 1
  fi
}

require_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $c" >&2
    exit 1
  fi
}

program_name() {
  case "$1" in
    proposed) echo "Hwimo_top030_proposed" ;;
    low030_mtl_all) echo "Hwimo_low030_mtl_all" ;;
    *) echo "[ERROR] unknown program key: $1" >&2; exit 1 ;;
  esac
}

p4_path() {
  case "$1" in
    proposed) echo "$ROOT/p4/hwimo_top030_proposed_bmv2.p4" ;;
    low030_mtl_all) echo "$ROOT/p4/hwimo_low030_mtl_all_bmv2.p4" ;;
    *) echo "[ERROR] unknown program key: $1" >&2; exit 1 ;;
  esac
}

build_json() {
  local name
  name="$(program_name "$1")"
  echo "$ROOT/build/${name}.json"
}

commands_path() {
  local name
  name="$(program_name "$1")"
  echo "$ROOT/commands/${name}_commands.txt"
}

expected_summary_path() {
  case "$1" in
    proposed) echo "$ROOT/expected/expected_Hwimo_top030_proposed_summary.json" ;;
    low030_mtl_all) echo "$ROOT/expected/expected_Hwimo_low030_mtl_all_summary.json" ;;
    *) echo "[ERROR] unknown program key: $1" >&2; exit 1 ;;
  esac
}

validate_program_arg() {
  case "$1" in
    proposed) echo "proposed" ;;
    low030_mtl_all) echo "low030_mtl_all" ;;
    *) echo "[ERROR] unknown program key: $1" >&2; exit 1 ;;
  esac
}

cleanup_switch() {
  if [[ -n "${SWITCH_PID:-}" ]]; then
    if kill -0 "$SWITCH_PID" >/dev/null 2>&1; then
      echo "[INFO] Stopping simple_switch pid=$SWITCH_PID"
      sudo kill "$SWITCH_PID" >/dev/null 2>&1 || true
      sleep 1
      sudo kill -9 "$SWITCH_PID" >/dev/null 2>&1 || true
    fi
    SWITCH_PID=""
  fi
}

cleanup_on_exit() {
  local status=$?
  if [[ "$KEEP_SWITCH" -eq 0 ]]; then
    cleanup_switch
  fi
  exit "$status"
}
trap cleanup_on_exit EXIT

check_environment() {
  echo "============================================================"
  echo "[CHECK] Environment"
  echo "============================================================"

  require_cmd sudo
  require_cmd ip
  require_cmd simple_switch
  require_cmd simple_switch_CLI
  require_cmd python3
  if [[ "$SKIP_COMPILE" -eq 0 ]]; then
    require_cmd p4c-bm2-ss
  fi

  require_file "$ROOT/scripts/validate_top030_bmv2_packet_replay_live.py"
  require_file "$ROOT/data/153_input.txt"
  require_file "$ROOT/data/test_meta.txt"
  require_file "$ROOT/data/task1_label_i.txt"

  # Make validator-compatible symlink if absent.
  ln -sf "$ROOT/data/153_input.txt" "$ROOT/data/test_input_153.txt"

  echo "ROOT=$ROOT"
  echo "TARGET=$TARGET"
  echo "LIMIT=$LIMIT"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "SEND_INTERVAL=$SEND_INTERVAL"
  echo "RETRY_SEND_INTERVAL=$RETRY_SEND_INTERVAL"
  echo "BATCH_TIMEOUT=$BATCH_TIMEOUT"
  echo "RETRIES=$RETRIES"
  echo "THRIFT_PORT=$THRIFT_PORT"
  echo "DEVICE_ID=$DEVICE_ID"
  echo "PACKET_PRINT=$PACKET_PRINT"
  echo "PACKET_PRINT_EVERY=$PACKET_PRINT_EVERY"
  echo "PACKET_PRINT_MAX=$PACKET_PRINT_MAX"

  echo
  echo "[CHECK] Input width/count"
  python3 - <<PY
from pathlib import Path
root = Path("$ROOT")
p = root / "data" / "test_input_153.txt"
widths = set()
n = 0
with open(p, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            widths.add(len(s))
            n += 1
print("input:", p)
print("line_count:", n)
print("widths:", sorted(widths))
if widths != {153}:
    raise SystemExit("[ERROR] Expected only width 153")
PY
}

compile_one() {
  local key="$1"
  local name p4 json log
  name="$(program_name "$key")"
  p4="$(p4_path "$key")"
  json="$(build_json "$key")"
  log="$ROOT/logs/compile_${name}.log"

  require_file "$p4"

  if [[ "$SKIP_COMPILE" -eq 1 ]]; then
    require_file "$json"
    echo "[SKIP] Compile $name"
    return
  fi

  echo "============================================================"
  echo "[COMPILE] $name"
  echo "============================================================"
  p4c-bm2-ss --p4v 16 -o "$json" "$p4" 2>&1 | tee "$log"

  if grep -qiE "error" "$log"; then
    echo "[ERROR] Compile log contains error: $log" >&2
    exit 1
  fi

  require_file "$json"
  ls -lh "$json"
}

setup_veth() {
  echo "============================================================"
  echo "[VETH] Clean setup"
  echo "============================================================"

  sudo pkill -9 -x simple_switch >/dev/null 2>&1 || true
  sleep 1
  # Clean stale BMv2 nanomsg notification sockets. A crashed BMv2 process can leave these behind.
  sudo rm -f /tmp/bmv2-*-notifications.ipc /tmp/bmv2-*-notifications.ipc.* 2>/dev/null || true
  sudo rm -f "/tmp/bmv2-${DEVICE_ID}-notifications.ipc" "/tmp/bmv2-${DEVICE_ID}-notifications.ipc.*" 2>/dev/null || true

  sudo ip link del veth0 2>/dev/null || true
  sudo ip link del veth2 2>/dev/null || true

  sudo ip link add veth0 type veth peer name veth1
  sudo ip link add veth2 type veth peer name veth3

  for i in veth0 veth1 veth2 veth3; do
    sudo ip link set dev "$i" up
    sudo ip link set dev "$i" promisc on
    sudo ip link set dev "$i" mtu 9500
    sudo ip link set dev "$i" txqueuelen 10000 || true
  done

  ip -br link | grep -E 'veth0|veth1|veth2|veth3'
}

wait_for_thrift() {
  local port="$1"
  local tries=30

  echo "[INFO] Waiting for simple_switch thrift port $port"
  for _ in $(seq 1 "$tries"); do
    if echo "show_tables" | simple_switch_CLI --thrift-port "$port" >/dev/null 2>&1; then
      echo "[OK] simple_switch_CLI connected"
      return 0
    fi
    sleep 1
  done

  echo "[ERROR] simple_switch thrift did not become ready" >&2
  return 1
}

start_switch() {
  local key="$1"
  local name json log
  name="$(program_name "$key")"
  json="$(build_json "$key")"
  log="$ROOT/logs/simple_switch_${name}.log"

  require_file "$json"

  echo "============================================================"
  echo "[SWITCH] Start $name"
  echo "============================================================"

  cleanup_switch || true
  sudo pkill -9 -x simple_switch >/dev/null 2>&1 || true
  sleep 1

  # A crashed BMv2 instance can leave stale nanomsg IPC sockets.
  sudo rm -f /tmp/bmv2-*-notifications.ipc /tmp/bmv2-*-notifications.ipc.* 2>/dev/null || true
  sudo rm -f "/tmp/bmv2-${DEVICE_ID}-notifications.ipc" "/tmp/bmv2-${DEVICE_ID}-notifications.ipc.*" 2>/dev/null || true

  rm -f "$log"

  # Do not pipe through tee here. Capture the actual sudo/simple_switch background process
  # and use the thrift server readiness test as the source of truth.
  sudo simple_switch \
    --device-id "$DEVICE_ID" \
    --log-console \
    -i 0@veth0 \
    -i 1@veth2 \
    --thrift-port "$THRIFT_PORT" \
    "$json" \
    > "$log" 2>&1 &
  SWITCH_PID=$!

  echo "[INFO] simple_switch pid=$SWITCH_PID"
  sleep 2

  if wait_for_thrift "$THRIFT_PORT"; then
    echo "[OK] simple_switch is ready"
    return 0
  fi

  echo "[ERROR] simple_switch thrift did not become ready. Process status:" >&2
  ps -fp "$SWITCH_PID" >&2 || true
  echo "[ERROR] Last 120 lines of log:" >&2
  tail -120 "$log" >&2 || true
  exit 1
}

load_rules() {
  local key="$1"
  local name cmd log
  name="$(program_name "$key")"
  cmd="$(commands_path "$key")"
  log="$ROOT/logs/load_${name}.log"

  require_file "$cmd"

  echo "============================================================"
  echo "[LOAD RULES] $name"
  echo "============================================================"

  sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$cmd" \
    | simple_switch_CLI --thrift-port "$THRIFT_PORT" \
    2>&1 | tee "$log"

  if grep -qiE "error|invalid|failed|unknown" "$log"; then
    echo "[ERROR] Rule load log contains error. See: $log" >&2
    grep -iE "error|invalid|failed|unknown" "$log" >&2 || true
    exit 1
  fi

  echo "[OK] Rules loaded: $cmd"
}

validate_one() {
  local key="$1"
  local name prog_arg log
  name="$(program_name "$key")"
  prog_arg="$(validate_program_arg "$key")"
  log="$ROOT/logs/validate_${name}.log"

  echo "============================================================"
  echo "[VALIDATE] $name"
  echo "============================================================"

  local live_args=()
  if [[ "$PACKET_PRINT" -eq 1 ]]; then
    live_args=(--live-print all --live-every "$PACKET_PRINT_EVERY" --live-max "$PACKET_PRINT_MAX")
  else
    live_args=(--live-print none)
  fi

  sudo python3 "$ROOT/scripts/validate_top030_bmv2_packet_replay_live.py" \
    --root "$ROOT" \
    --program "$prog_arg" \
    --iface-in veth1 \
    --iface-out veth3 \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --send-interval "$SEND_INTERVAL" \
    --retry-send-interval "$RETRY_SEND_INTERVAL" \
    --batch-timeout "$BATCH_TIMEOUT" \
    --retries "$RETRIES" \
    "${live_args[@]}" \
    2>&1 | tee "$log"

  if grep -qiE "Traceback|OSError|No such device|ERROR" "$log"; then
    echo "[ERROR] Validation log contains an error. See: $log" >&2
    exit 1
  fi

  echo "[OK] Validation finished: $log"
}
print_accuracy_reference() {
  local key="$1"
  local summary
  summary="$(expected_summary_path "$key")"
  if [[ -f "$summary" ]]; then
    echo "============================================================"
    echo "[REFERENCE EXPECTED SUMMARY] $(program_name "$key")"
    echo "============================================================"
    python3 - <<PY
import json
from pathlib import Path
p = Path("$summary")
j = json.load(open(p))
print("program:", j.get("program"))
print("macro_accuracy:", j.get("macro_accuracy"))
for k, v in j.get("accuracy", {}).items():
    print(k, v.get("accuracy"))
PY
  fi
}

run_one() {
  local key="$1"
  local name
  name="$(program_name "$key")"

  echo
  echo "################################################################################"
  echo "# RUN $name"
  echo "################################################################################"

  compile_one "$key"
  setup_veth
  start_switch "$key"
  load_rules "$key"
  validate_one "$key"
  print_accuracy_reference "$key"

  if [[ "$KEEP_SWITCH" -eq 0 ]]; then
    cleanup_switch
  fi
}

check_environment
sudo -v

case "$TARGET" in
  proposed)
    run_one proposed
    ;;
  low030_mtl_all)
    run_one low030_mtl_all
    ;;
  both)
    run_one proposed
    run_one low030_mtl_all
    ;;
  *)
    echo "[ERROR] Unknown target: $TARGET" >&2
    exit 1
    ;;
esac

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"
echo "Result summaries:"
find "$ROOT/results" -maxdepth 1 -type f -name '*packet_validation_summary.json' -printf '  %p\n' | sort
