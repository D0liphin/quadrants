#!/bin/bash

# Diagnostics branch: use `set -x` only (no `-e`) so we always collect the timing
# summary and the resource-sampler log even if some tests fail (e.g. when the
# prebuilt wheel is slightly out of sync with the checked-out test tree).
set -x

export QD_FILE_TIMING=1
export QD_FILE_TIMING_OUTPUT="${RUNNER_TEMP}/file_timing.md"

# --- background resource sampler -------------------------------------------------
# The slowdown looks like a system-wide stall on certain runner VMs, so sample
# system memory/swap/load + top-RSS processes every 15s into an artifact. We track
# system totals (not just parent RSS) because pytest-xdist runs up to 8 workers.
RES_LOG="${RUNNER_TEMP}/resource_sampler.log"
(
  while true; do
    echo "==== $(date -u +%H:%M:%S)Z ===="
    sysctl -n vm.swapusage
    vm_stat
    uptime
    ps -A -o rss,pid,comm | sort -rn | head -n 8
    echo
    sleep 15
  done
) > "${RES_LOG}" 2>&1 &
SAMPLER_PID=$!
trap 'kill "${SAMPLER_PID}" 2>/dev/null || true' EXIT

pip install --prefer-binary --group test
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"

# The C++ test binary is a build artifact; it won't exist when installing a prebuilt
# wheel, so only run it if present.
if [ -x ./build/quadrants_cpp_tests ]; then
  ./build/quadrants_cpp_tests
fi

# Phase 1: run all tests except torch-dependent ones.
# `/usr/bin/time -l` reports the parent-process peak RSS on its stderr (into the job
# log); the sampler above captures the system-wide picture across all workers.
/usr/bin/time -l python tests/run_tests.py -v -r 1 --arch metal,vulkan,cpu -m "not needs_torch"

# Phase 2: install torch, run only torch tests
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
/usr/bin/time -l python tests/run_tests.py -v -r 1 --arch metal,vulkan,cpu -m needs_torch

if [ -f "$QD_FILE_TIMING_OUTPUT" ]; then
  cat "$QD_FILE_TIMING_OUTPUT" >> "$GITHUB_STEP_SUMMARY"
fi

# Surface the tail of the sampler in the job summary for quick triage.
{
  echo ""
  echo "### Resource sampler (tail of resource_sampler.log)"
  echo '```'
  tail -n 60 "${RES_LOG}"
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"
