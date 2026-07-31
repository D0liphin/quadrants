#!/bin/bash
# [DONOTMERGE] Capture the EXACT per-cycle Vulkan call sequence MoltenVK sees during qd.init()/qd.reset(),
# so a standalone (no-Quadrants) C++ reproducer can mirror it 1:1. MoltenVK logs every Vulkan entry point
# when MVK_CONFIG_TRACE_VULKAN_CALLS is set. We run 3 cycles with stderr markers, then summarize the
# call multiset for a "steady-state" cycle (cycle 1, past first-time instance/volk init one-offs).
set -x
OUT="${RUNNER_TEMP}/vktrace"; mkdir -p "$OUT"
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
# 1 = log each Vulkan call on entry. (Higher values add exit/duration; entry is enough for the sequence.)
export MVK_CONFIG_TRACE_VULKAN_CALLS=1

python - 2> "$OUT/trace.txt" <<'PY'
import sys
import quadrants as qd
for i in range(3):
    sys.stderr.write("=== CYCLE %d INIT ===\n" % i); sys.stderr.flush()
    qd.init(arch=qd.vulkan)
    sys.stderr.write("=== CYCLE %d RESET ===\n" % i); sys.stderr.flush()
    qd.reset()
sys.stderr.write("=== DONE ===\n"); sys.stderr.flush()
PY

echo "trace total lines: $(wc -l < "$OUT/trace.txt")"

# Extract the calls of ONE steady-state init and ONE steady-state reset (cycle 1) into ordered lists.
awk '/=== CYCLE 1 INIT ===/{c=1;next} /=== CYCLE 1 RESET ===/{c=2} /=== CYCLE 2 INIT ===/{c=0}
     c==1{print}' "$OUT/trace.txt" | grep -oaE 'vk[A-Za-z]+' > "$OUT/init_calls.txt"
awk '/=== CYCLE 1 RESET ===/{c=1;next} /=== CYCLE 2 INIT ===/{c=0} c==1{print}' "$OUT/trace.txt" \
     | grep -oaE 'vk[A-Za-z]+' > "$OUT/reset_calls.txt"

{
  echo "## Vulkan call trace: one steady-state qd.init + qd.reset (cycle 1)"
  echo ""
  echo "### init call multiset (count x function)"
  echo '```'
  sort "$OUT/init_calls.txt" | uniq -c | sort -rn
  echo '```'
  echo "### reset call multiset"
  echo '```'
  sort "$OUT/reset_calls.txt" | uniq -c | sort -rn
  echo '```'
  echo "### init calls in ORDER (first 120)"
  echo '```'
  head -120 "$OUT/init_calls.txt"
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"
