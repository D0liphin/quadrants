#!/bin/bash

# Within-VM A/B for the Mac teardown collapse.
#
# Every previous experiment compared a "fix" leg against a "control" leg on SEPARATE
# runner VMs. But the baseline is bimodal (the same code is ~50 min on a good VM and
# 150-350 min on a bad one), so a single fix-vs-control pair cannot separate "the fix
# helped" from "the fix leg drew a good VM". Here we run BOTH arms back-to-back in ONE
# job (one VM), so the VM factor cancels and any wall-time delta is attributable to the
# arm itself.
#
# Arm B (candidate): `--forked` runs each test in its own forked subprocess, so teardown
# is process exit instead of the in-process qd.reset() accumulation the collapse lives
# in. genesis-world runs `--forked` and never collapses.
# Arm A (baseline): the current in-process invocation (reproduces the collapse).
#
# We run the candidate FIRST so its (expected fast) result is always captured even if the
# baseline arm later collapses and runs long; the two pytest sessions are independent
# (fresh offline cache per invocation) so order does not contaminate the comparison.

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QD_FILE_TIMING=1
export QD_FILE_TIMING_OUTPUT="${RUNNER_TEMP}/file_timing.md"

ARCHES="metal,vulkan,cpu"
SUMMARY="${RUNNER_TEMP}/withinvm_summary.txt"
: > "${SUMMARY}"

# --- background resource sampler + load-triggered deep capture (same as 4_test.sh) ----
RES_LOG="${RUNNER_TEMP}/resource_sampler.log"
DEEP_DIR="${RUNNER_TEMP}/deep_capture"
mkdir -p "${DEEP_DIR}"
LOAD_THRESH=80
DEEP_MAX=12
DEEP_MIN_GAP=90
(
  deep_n=0
  last_deep=0
  while true; do
    now_epoch=$(date +%s)
    load1=$(sysctl -n vm.loadavg | awk '{print $2}')
    nthreads=$(ps -M -A 2>/dev/null | wc -l | tr -d ' ')
    nproc=$(ps -A 2>/dev/null | wc -l | tr -d ' ')
    echo "==== $(date -u +%H:%M:%S)Z load1=${load1} procs=${nproc} threads=${nthreads} ===="
    sysctl -n vm.swapusage
    vm_stat
    uptime
    ps -A -o rss,pid,stat,comm | sort -rn | head -n 8
    echo
    if [ "${deep_n}" -lt "${DEEP_MAX}" ] \
       && awk "BEGIN{exit !(${load1:-0} > ${LOAD_THRESH})}" \
       && [ $((now_epoch - last_deep)) -ge "${DEEP_MIN_GAP}" ]; then
      ts=$(date -u +%H%M%S)
      df="${DEEP_DIR}/deep_${ts}_load${load1}.txt"
      {
        echo "=== DEEP CAPTURE ${ts}Z load1=${load1} threads=${nthreads} ==="
        echo "--- process STAT summary (first char) ---"
        ps -A -o stat,comm | awk 'NR>1{print substr($1,1,1)}' | sort | uniq -c | sort -rn
        echo "--- ps: pid ppid stat wchan %cpu rss comm ---"
        ps -A -o pid,ppid,stat,wchan,%cpu,rss,comm
        echo "--- ps -M -A (all threads, raw) ---"
        ps -M -A
      } > "${df}" 2>&1
      sudo -n /usr/sbin/spindump -notarget 3 10 -o "${DEEP_DIR}/spindump_${ts}.txt" >/dev/null 2>&1 \
        || echo "spindump skipped ${ts}" >> "${df}"
      deep_n=$((deep_n+1))
      last_deep=${now_epoch}
    fi
    sleep 15
  done
) > "${RES_LOG}" 2>&1 &
SAMPLER_PID=$!
cleanup() { kill "${SAMPLER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

pip install --prefer-binary --group test
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
# pytest-forked provides the `--forked` flag used by the candidate arm.
pip install pytest-forked
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"

run_phase() {
  local label="$1"; shift
  local t0 t1 rc
  t0=$(date +%s)
  echo "==== WITHINVM PHASE ${label} START $(date -u +%H:%M:%SZ) (epoch ${t0}) ===="
  echo "==== WITHINVM PHASE ${label} START $(date -u +%H:%M:%SZ) ====" >> "${RES_LOG}"
  "$@"
  rc=$?
  t1=$(date +%s)
  echo "==== WITHINVM PHASE ${label} END rc=${rc} elapsed=$(( t1 - t0 ))s ===="
  echo "==== WITHINVM PHASE ${label} END rc=${rc} elapsed=$(( t1 - t0 ))s ====" >> "${RES_LOG}"
  echo "WITHINVM_RESULT ${label} elapsed_s=$(( t1 - t0 )) rc=${rc}" >> "${SUMMARY}"
}

# Candidate first: each test forked into its own subprocess (teardown == process exit).
run_phase forked env QD_EXTRA_PYTEST_ARGS="--forked" \
  /usr/bin/time -l python tests/run_tests.py -v -r 1 --arch "${ARCHES}" -m "not needs_torch"

# Rename the per-phase file-timing so the baseline arm's does not overwrite it.
[ -f "${QD_FILE_TIMING_OUTPUT}" ] && mv "${QD_FILE_TIMING_OUTPUT}" "${RUNNER_TEMP}/file_timing_forked.md"

# Baseline: current in-process invocation (reproduces the collapse on a bad VM).
run_phase baseline env -u QD_EXTRA_PYTEST_ARGS \
  /usr/bin/time -l python tests/run_tests.py -v -r 1 --arch "${ARCHES}" -m "not needs_torch"

[ -f "${QD_FILE_TIMING_OUTPUT}" ] && mv "${QD_FILE_TIMING_OUTPUT}" "${RUNNER_TEMP}/file_timing_baseline.md"

# --- job summary --------------------------------------------------------------------
{
  echo "## Within-VM A/B: --forked (candidate) vs in-process (baseline)"
  echo "Both arms ran the same 'not needs_torch' phase-1 workload (arch=${ARCHES}) on the"
  echo "SAME runner VM, so runner-lottery variance is cancelled."
  echo ""
  echo '```'
  cat "${SUMMARY}"
  echo '```'
  echo ""
  echo "### Resource sampler (tail)"
  echo '```'
  tail -n 80 "${RES_LOG}"
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"
