#!/bin/bash
# [DONOTMERGE] Leak-localization probe for the vulkan/MoltenVK per-cycle leak.
#
# Established: RSS grows ~50 KB per qd.init()/qd.reset() cycle (linear, no plateau) and MoltenVK
# eventually returns VK_ERROR_DEVICE_LOST. This splits that leak across three layers by running the
# SAME init/reset loop with increasing work, each in its own fresh process, logging ru_maxrss/cycle:
#   bare   : qd.init(vulkan) -> qd.reset()                       (device + GfxRuntime root buffers only)
#   field  : + allocate a field + to_numpy (readback)           (adds buffer/memory alloc + submit)
#   kernel : + compile & launch a FRESH kernel each cycle        (adds pipeline/shader compilation)
# Comparing the per-cycle slopes localizes the leak to device- / buffer- / pipeline-level.

set -x
OUT="${RUNNER_TEMP}/leak_out"; mkdir -p "$OUT"
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
export MVK_CONFIG_LOG_LEVEL=1
echo "python=$(python -V 2>&1) quadrants=$(python -c 'import quadrants as qd; print(getattr(qd,"__version__","?"))' 2>&1)"

# macOS has no GNU timeout; portable watchdog shim (background + TERM/KILL after N seconds).
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

PROBE_PY="${RUNNER_TEMP}/leak_probe.py"
cat > "$PROBE_PY" <<'PY'
import os, resource
import quadrants as qd

MODE = os.environ.get("LEAK_MODE", "kernel")
N = int(os.environ.get("LEAK_CYCLES", "4000"))
LOG = os.environ.get("LEAK_LOG", "/tmp/leak.txt")
open(LOG, "w").close()


def rss_kb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 1024 if r > 10_000_000 else r


print("LEAK probe mode=%s cycles=%d" % (MODE, N), flush=True)
i = 0
try:
    for i in range(N):
        qd.init(arch=qd.vulkan)
        if MODE in ("field", "kernel"):
            m = 256 + (i % 128)
            fld = qd.field(dtype=qd.f32, shape=m)
            if MODE == "kernel":
                @qd.kernel
                def kern(s: qd.f32):
                    for j in range(m):
                        fld[j] = s * j

                kern(1.0 + (i % 5))
            _ = fld.to_numpy()  # readback forces a submit + device sync
        qd.reset()
        if i % 25 == 0:
            with open(LOG, "a") as fh:
                fh.write("cycle=%d max_rss_kb=%d\n" % (i, rss_kb()))
    print("LEAK_DONE mode=%s ran=%d final_max_rss_kb=%d" % (MODE, N, rss_kb()), flush=True)
except BaseException as e:
    print("LEAK_THREW mode=%s cycle=%d %s: %s" % (MODE, i, type(e).__name__, str(e)[:160]), flush=True)
    with open(LOG, "a") as fh:
        fh.write("THREW cycle=%d %s\n" % (i, type(e).__name__))
    raise
PY

run_probe() {  # run_probe NAME MODE SECS CYCLES
  local name="$1" mode="$2" secs="$3" cyc="$4"
  echo "############### PROBE ${name} (mode=${mode}) START $(date -u +%H:%M:%SZ) ###############"
  run_to "$secs" env LEAK_MODE="$mode" LEAK_CYCLES="$cyc" LEAK_LOG="$OUT/${name}.txt" python "$PROBE_PY"
  echo "############### PROBE ${name} END rc=$? $(date -u +%H:%M:%SZ) ###############"
}

# Cycle target well above the old ~2900 device-lost cap; per-stage timeouts give headroom for the
# (now costlier) per-cycle instance recreate. PASS if a probe runs past ~2900 with NO rc=134 abort.
run_probe bare   bare   700 5000
run_probe field  field  700 5000
run_probe kernel kernel 800 5000

# --- summary: per-cycle slope for each layer --------------------------------------------
{
  echo "## Leak localization: RSS slope by layer"
  echo ""
  echo "bare = init/reset only (device + runtime buffers); field = +alloc+readback; kernel = +fresh pipeline/launch."
  echo "A near-zero slope for a layer means that layer does NOT leak; the layer where the slope jumps is the culprit."
  echo ""
  for f in bare field kernel; do
    echo "### ${f}"
    echo '```'
    if [ -s "$OUT/$f.txt" ]; then
      head -1 "$OUT/$f.txt"
      awk 'NR%20==0' "$OUT/$f.txt" | tail -6
      echo "-- last line --"; tail -1 "$OUT/$f.txt"
      awk -F'[= ]' '/^cycle=/{c=$2; r=$4; if(f0==0){c0=c; r0=r; f0=1}} END{if(c>c0) printf "SLOPE %s = %.2f kb/cycle  (%d -> %d kb over %d cycles)\n","'"$f"'",(r-r0)/(c-c0),r0,r,c-c0}' "$OUT/$f.txt"
    else
      echo "(no data)"
    fi
    echo '```'
  done
} >> "$GITHUB_STEP_SUMMARY"
