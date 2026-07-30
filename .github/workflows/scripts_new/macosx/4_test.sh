#!/bin/bash

# Diagnostics branch: use `set -x` only (no `-e`) so we always collect the timing
# summary and the resource-sampler log even if some tests fail (e.g. when the
# prebuilt wheel is slightly out of sync with the checked-out test tree).
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# QD_FIX selects the CPU-contention mitigation under test (all hang-free, unlike the
# earlier SIGSTOP quiesce which blocked synchronous XPC callers):
#   none    - control: change nothing (baseline slow run).
#   deprio  - keep the consumer daemons alive but drop them to the background CPU/IO
#             tier (renice +20 + taskpolicy -b) so they yield the 3 cores to pytest.
#   disable - launchctl disable + bootout the consumer daemons (fail-fast: callers get
#             "service unavailable" instead of hanging).
QD_FIX="${QD_FIX:-none}"
# SPLIT_ARCH selects which backend(s) this leg tests. Split per-backend (metal / vulkan
# / cpu) in the matrix so we can see whether the teardown stall is arch-specific (e.g.
# Metal GPU-context teardown) or backend-independent. Defaults to all three.
# NB: deliberately NOT named QD_ARCH - that is a reserved quadrants env var that makes
# qd.init() override its arch via arch_from_name(), which aborts on the alias "cpu".
SPLIT_ARCH="${SPLIT_ARCH:-metal,vulkan,cpu}"
# Sentinel: the sampler's periodic re-apply must NOT touch any daemon until this file
# exists, which happens only after all pip installs (network daemons must stay usable
# for downloads). Without the gate we would throttle/remove networking mid `pip
# install`.
FIX_READY="${RUNNER_TEMP}/fix_ready.flag"
rm -f "${FIX_READY}"

export QD_FILE_TIMING=1
export QD_FILE_TIMING_OUTPUT="${RUNNER_TEMP}/file_timing.md"

# --- background resource sampler + load-triggered deep capture -------------------
# The slow legs show the 1-min load average spiking to ~400 with 0 swap, entirely in
# the teardown phase. The 15s sampler records the system picture (mem/swap/load +
# process count + total thread count). When load1 crosses LOAD_THRESH we additionally
# dump *what* the threads are doing (per-process STAT/wchan, per-worker fd + thread
# counts, `sample` stacks of the top workers, and a system `spindump`) so we can tell a
# thread/fd leak from driver-blocked (uninterruptible) threads. All go to artifacts.
RES_LOG="${RUNNER_TEMP}/resource_sampler.log"
DEEP_DIR="${RUNNER_TEMP}/deep_capture"
mkdir -p "${DEEP_DIR}"
LOAD_THRESH=80      # baseline load is single-digit..~40; the spike is ~400
DEEP_MAX=10         # cap deep captures (spindump files are large)
DEEP_MIN_GAP=90     # min seconds between deep captures
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
    # Re-apply the mitigation periodically to catch daemons launchd relaunched or that
    # just woke from a timer. Gated on the sentinel so we never touch networking before
    # pip installs have finished.
    if [ -f "${FIX_READY}" ]; then
      case "${QD_FIX}" in
        deprio)  bash "${SCRIPT_DIR}/deprio_daemons.sh"  >/dev/null 2>&1 || true ;;
        disable) bash "${SCRIPT_DIR}/disable_daemons.sh" >/dev/null 2>&1 || true ;;
      esac
    fi
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
        echo "--- per python worker: threads + fds ---"
        for p in $(pgrep python 2>/dev/null); do
          echo "pid=$p threads=$(ps -M -p "$p" 2>/dev/null | tail -n +2 | wc -l | tr -d ' ') fds=$(lsof -p "$p" 2>/dev/null | wc -l | tr -d ' ')"
        done
        echo "--- ps -M -A (all threads, raw) ---"
        ps -M -A
      } > "${df}" 2>&1
      # per-worker user-space stack samples (no sudo needed for own processes)
      for p in $(ps -A -o pid,rss,comm | grep -i python | sort -k2 -rn | head -3 | awk '{print $1}'); do
        sample "$p" 3 -file "${DEEP_DIR}/sample_${ts}_pid${p}.txt" >/dev/null 2>&1 &
      done
      # system-wide spindump (best effort; needs passwordless sudo)
      sudo -n /usr/sbin/spindump -notarget 3 10 -o "${DEEP_DIR}/spindump_${ts}.txt" >/dev/null 2>&1 \
        || echo "spindump skipped ${ts}" >> "${df}"
      deep_n=$((deep_n+1))
      last_deep=${now_epoch}
    fi
    sleep 15
  done
) > "${RES_LOG}" 2>&1 &
SAMPLER_PID=$!
cleanup() {
  kill "${SAMPLER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

pip install --prefer-binary --group test
# Install torch up front (needs network). Kept ahead of the mitigation step below so
# the network daemons are still up/usable during downloads; the only difference between
# the control and fix legs is the mitigation itself.
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"

# Apply the mitigation now that all downloads are done, then let the sampler re-apply it
# periodically (gated on the sentinel).
case "${QD_FIX}" in
  deprio)  bash "${SCRIPT_DIR}/deprio_daemons.sh"  | tee -a "$GITHUB_STEP_SUMMARY" ;;
  disable) bash "${SCRIPT_DIR}/disable_daemons.sh" | tee -a "$GITHUB_STEP_SUMMARY" ;;
  none)    echo "QD_FIX=none (control): no CPU-contention mitigation applied" ;;
esac
touch "${FIX_READY}"

# The C++ test binary is a build artifact; it won't exist when installing a prebuilt
# wheel, so only run it if present.
if [ -x ./build/quadrants_cpp_tests ]; then
  ./build/quadrants_cpp_tests
fi

# Phase 1: run all tests except torch-dependent ones.
# `/usr/bin/time -l` reports the parent-process peak RSS on its stderr (into the job
# log); the sampler above captures the system-wide picture across all workers.
/usr/bin/time -l python tests/run_tests.py -v -r 1 --arch "${SPLIT_ARCH}" -m "not needs_torch"

# Phase 2: torch was installed above; run only torch tests.
/usr/bin/time -l python tests/run_tests.py -v -r 1 --arch "${SPLIT_ARCH}" -m needs_torch

if [ -f "$QD_FILE_TIMING_OUTPUT" ]; then
  cat "$QD_FILE_TIMING_OUTPUT" >> "$GITHUB_STEP_SUMMARY"
fi

# Surface the tail of the sampler + deep-capture STAT summaries in the job summary.
{
  echo ""
  echo "### Resource sampler (tail of resource_sampler.log)"
  echo '```'
  tail -n 60 "${RES_LOG}"
  echo '```'
  echo ""
  echo "### Deep captures (load > ${LOAD_THRESH})"
  ls -1 "${DEEP_DIR}" 2>/dev/null || echo "(none - load never crossed threshold)"
  echo ""
  echo "Process STAT summaries at each spike:"
  echo '```'
  for f in "${DEEP_DIR}"/deep_*.txt; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f") =="
    sed -n '/process STAT summary/,/ps: pid/p' "$f" | sed '$d'
  done
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"
