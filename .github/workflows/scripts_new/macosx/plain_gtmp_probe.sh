#!/bin/bash
# [DONOTMERGE] Validate Darwin non-VMA gtmp/listgen against the MoltenVK hang.
#
# Baseline (stock VMA for gtmp+listgen): bare qd.init(vulkan)/qd.reset() aborts ~2900 cycles
# with kIOGPUCommandBufferCallbackErrorHang / VK_ERROR_DEVICE_LOST.
# Expectation with bypass_pooled_allocator on those two buffers: LEAK_DONE at 5000, rc=0.

set -x
OUT="${RUNNER_TEMP}/plain_gtmp_out"; mkdir -p "$OUT"
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
export MVK_CONFIG_LOG_LEVEL=1
echo "python=$(python -V 2>&1) quadrants=$(python -c 'import quadrants as qd; print(getattr(qd,"__version__","?"))' 2>&1)"

run_to() {
  local secs="$1"; shift
  "$@" &
  local cmd_pid=$!
  ( sleep "$secs"; kill -TERM "$cmd_pid" 2>/dev/null && { sleep 8; kill -KILL "$cmd_pid" 2>/dev/null; } ) &
  local wd_pid=$!
  local rc=0
  wait "$cmd_pid" 2>/dev/null || rc=$?
  kill "$wd_pid" 2>/dev/null; pkill -P "$wd_pid" 2>/dev/null; wait "$wd_pid" 2>/dev/null || true
  return "$rc"
}

PROBE_PY="${RUNNER_TEMP}/plain_gtmp_probe.py"
cat > "$PROBE_PY" <<'PY'
import os, resource
import quadrants as qd

N = int(os.environ.get("LEAK_CYCLES", "5000"))
LOG = os.environ.get("LEAK_LOG", "/tmp/leak.txt")
open(LOG, "w").close()


def rss_kb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 1024 if r > 10_000_000 else r


print("PLAIN_GTMP bare probe cycles=%d" % N, flush=True)
i = 0
try:
    for i in range(N):
        qd.init(arch=qd.vulkan)
        qd.reset()
        if i % 25 == 0:
            with open(LOG, "a") as fh:
                fh.write("cycle=%d max_rss_kb=%d\n" % (i, rss_kb()))
    print("LEAK_DONE mode=bare ran=%d final_max_rss_kb=%d" % (N, rss_kb()), flush=True)
except BaseException as e:
    print("LEAK_THREW mode=bare cycle=%d %s: %s" % (i, type(e).__name__, str(e)[:160]), flush=True)
    with open(LOG, "a") as fh:
        fh.write("THREW cycle=%d %s\n" % (i, type(e).__name__))
    raise
PY

echo "############### PROBE bare START $(date -u +%H:%M:%SZ) ###############"
run_to 900 env LEAK_CYCLES=5000 LEAK_LOG="$OUT/bare.txt" python "$PROBE_PY"
RC=$?
echo "############### PROBE bare END rc=${RC} $(date -u +%H:%M:%SZ) ###############"

{
  echo "## Darwin plain gtmp/listgen bare probe"
  echo ""
  echo "PASS = \`LEAK_DONE\` at 5000 with rc=0 (past the ~2900 VMA hang). FAIL = abort/device-lost earlier."
  echo ""
  echo "### bare rc=${RC}"
  echo '```'
  if [ -s "$OUT/bare.txt" ]; then
    head -1 "$OUT/bare.txt"
    awk 'NR%20==0' "$OUT/bare.txt" | tail -6
    echo "-- last line --"; tail -1 "$OUT/bare.txt"
    awk -F'[= ]' '/^cycle=/{c=$2; r=$4; if(f0==0){c0=c; r0=r; f0=1}} END{if(c>c0) printf "SLOPE bare = %.2f kb/cycle  (%d -> %d kb over %d cycles)\n",(r-r0)/(c-c0),r0,r,c-c0}' "$OUT/bare.txt"
  else
    echo "(no data)"
  fi
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"

exit "$RC"
