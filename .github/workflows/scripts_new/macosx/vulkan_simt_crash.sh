#!/bin/bash

# [DONOTMERGE] diagnostic branch. Dig into the vulkan SIMT subgroup crash seen when the
# per-backend-split vulkan leg is run with a single xdist worker (QD_TEST_THREADS=1):
#
#   tests/python/test_simt.py::test_subgroup_inclusive_mul_tiled[arch=vulkan-dtype3-5]
#   -> Abort trap: 6 (SIGABRT / process exit 134) at ~63% of the suite.
#
# dtype index 3 in _SCENARIOS_I32_AND_FLOATS is qd.f64, so the crashing case is the f64
# inclusive prefix-product on vulkan/MoltenVK. _check_inclusive_scan() calls
# _skip_if_f64_unsupported(), which is SUPPOSED to pytest.skip f64 on vulkan+Darwin
# ("MoltenVK does not support f64") - and it DID skip under 2 xdist workers, but under a
# single serial worker the test ran and aborted. This job isolates whether:
#   (a) the f64 subgroup kernel itself hard-aborts MoltenVK (test/backend bug), and/or
#   (b) the _skip_if_f64_unsupported guard fails to fire in the serial/multi-arch path.
# It captures the native backtrace (lldb + macOS .ips crash reports) so we can see the
# faulting frame (MoltenVK / Metal / codegen).

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CRASH_OUT="${RUNNER_TEMP}/crash_reports"
mkdir -p "${CRASH_OUT}"
STAGE_RESULTS="${RUNNER_TEMP}/stage_results.txt"
: > "${STAGE_RESULTS}"

DR_USER="${HOME}/Library/Logs/DiagnosticReports"
DR_SYS="/Library/Logs/DiagnosticReports"
mkdir -p "${DR_USER}"
# Clear any pre-existing crash reports so we only collect the ones this job produces.
rm -f "${DR_USER}"/*.ips "${DR_USER}"/*.crash 2>/dev/null || true
sudo -n rm -f "${DR_SYS}"/*.ips "${DR_SYS}"/*.crash 2>/dev/null || true

# Give MoltenVK / Metal a chance to log what it was doing right before it aborts.
export MVK_CONFIG_LOG_LEVEL=3
export MVK_DEBUG=1
export MVK_CONFIG_DEBUG=1
export MTL_DEBUG_LAYER=1

# .ips crash reports are JSON (header line + body doc). Extract the exception, the
# termination reason and the faulting thread's frames (with image names).
IPS_PARSER="${RUNNER_TEMP}/parse_ips.py"
cat > "${IPS_PARSER}" <<'PY'
import json, sys
path = sys.argv[1]
raw = open(path, "r", errors="replace").read()
parts = raw.split("\n", 1)
try:
    body = json.loads(parts[1]) if len(parts) > 1 else json.loads(raw)
except Exception as e:
    print("  (could not parse .ips as JSON: %s)" % e)
    print(raw[:4000])
    sys.exit(0)
print("  procName:", body.get("procName"))
print("  exception:", body.get("exception"))
print("  termination:", body.get("termination"))
print("  asi:", body.get("asi"))
images = body.get("usedImages", [])
def img(i):
    try:
        return images[i].get("name") or images[i].get("path", "?")
    except Exception:
        return "img%s" % i
for t in body.get("threads", []):
    if t.get("triggered"):
        print("  --- faulting thread %s (%s) ---" % (t.get("id"), t.get("name", "")))
        for fr in t.get("frames", [])[:40]:
            print("    %-28s %s +%s" % (img(fr.get("imageIndex", -1)),
                                        fr.get("symbol", ""),
                                        fr.get("symbolLocation", "")))
        break
PY

collect_and_dump_crashes() {
  local tag="$1"
  local found=0
  for d in "${DR_USER}" "${DR_SYS}"; do
    for f in "${d}"/*.ips "${d}"/*.crash; do
      [ -f "${f}" ] || continue
      local dest="${CRASH_OUT}/${tag}__$(basename "${f}")"
      cp "${f}" "${dest}" 2>/dev/null || sudo -n cp "${f}" "${dest}" 2>/dev/null || continue
      found=1
      echo "=== CRASH REPORT (${tag}): $(basename "${f}") ==="
      case "${f}" in
        *.ips)   python "${IPS_PARSER}" "${dest}" 2>&1 ;;
        *.crash) grep -E "Exception Type|Termination|Crashed Thread|^Thread [0-9]+ Crashed" "${dest}" | head -20 ;;
      esac
    done
  done
  # Move consumed reports aside so the next stage starts clean.
  rm -f "${DR_USER}"/*.ips "${DR_USER}"/*.crash 2>/dev/null || true
  sudo -n rm -f "${DR_SYS}"/*.ips "${DR_SYS}"/*.crash 2>/dev/null || true
  [ "${found}" -eq 1 ] || echo "(no crash report produced for ${tag})"
}

run_stage() {
  local label="$1"; shift
  echo "############### STAGE ${label} START $(date -u +%H:%M:%SZ) ###############"
  ( "$@" ); local rc=$?
  echo "############### STAGE ${label} END rc=${rc} $(date -u +%H:%M:%SZ) ###############"
  # SIGABRT surfaces as rc 134; give the crash reporter a moment to write the .ips.
  sleep 5
  collect_and_dump_crashes "${label}"
  echo "STAGE_RESULT ${label} rc=${rc}" >> "${STAGE_RESULTS}"
}

pip install --prefer-binary --group test
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
echo "python=$(python -V 2>&1) quadrants=$(python -c 'import quadrants as qd; print(getattr(qd,"__version__","?"))' 2>&1)"

# --- Stage 1: does plain (non-subgroup) f64 field I/O abort on vulkan? -----------------
# Tests the broad claim in the skip message ("MoltenVK does not support f64"). If this
# aborts, f64 is fundamentally unusable on vulkan/Darwin; if it prints OK, only the
# subgroup prefix-scan path is the problem.
PROBE="${RUNNER_TEMP}/probe_f64.py"
cat > "${PROBE}" <<'PY'
import platform
import quadrants as qd
qd.init(arch=qd.vulkan)
cfg = qd.lang.impl.current_cfg()
print("PROBE cfg.arch=", cfg.arch, "qd.vulkan=", qd.vulkan, "Darwin=", platform.system() == "Darwin")
f = qd.field(dtype=qd.f64, shape=8)
@qd.kernel
def fill():
    for i in range(8):
        f[i] = 1.5 * i
fill()
print("PROBE f64 basic field I/O OK:", f.to_numpy())
PY
run_stage probe_f64_basic python "${PROBE}"

# --- Stage 2: the exact crashing case, in isolation, fresh serial process --------------
# If this SKIPS -> the guard works standalone and the crash needs prior-test accumulation
# (or an arch-detection leak). If it ABORTS -> the f64 subgroup kernel itself crashes.
run_stage isolate_f64_mul \
  python tests/run_tests.py test_simt -k "test_subgroup_inclusive_mul_tiled and dtype3" --arch vulkan -t 1 -v -s

# --- Stage 3: all inclusive-scan tiled ops (add/mul/min/max), serial, under lldb -------
# add/min/max also go through _check_inclusive_scan; confirm whether they skip f64 or
# also abort. lldb prints the faulting backtrace directly into the job log.
run_stage inclusive_scan_lldb \
  lldb --batch -o "run" -k "thread backtrace all" -k "quit" -- \
  python tests/run_tests.py test_simt -k "inclusive_add_tiled or inclusive_mul_tiled or inclusive_min_tiled or inclusive_max_tiled" --arch vulkan -t 1 -v

# --- Stage 4: full-suite serial repro under lldb (only if isolation didn't already crash)
if ls "${CRASH_OUT}"/isolate_f64_mul__* "${CRASH_OUT}"/inclusive_scan_lldb__* >/dev/null 2>&1; then
  echo "an earlier stage already reproduced the crash; skipping full-suite repro"
  echo "STAGE_RESULT repro_simt_full skipped" >> "${STAGE_RESULTS}"
else
  run_stage repro_simt_full \
    lldb --batch -o "run" -k "thread backtrace all" -k "quit" -- \
    python tests/run_tests.py test_simt --arch vulkan -t 1 -v
fi

# --- job summary -----------------------------------------------------------------------
{
  echo "## Vulkan SIMT subgroup crash diagnostics"
  echo ""
  echo "Crashing case: test_subgroup_inclusive_mul_tiled[arch=vulkan, dtype=f64] (Abort trap: 6)."
  echo ""
  echo '```'
  cat "${STAGE_RESULTS}"
  echo '```'
  echo ""
  echo "### Crash reports collected"
  ls -1 "${CRASH_OUT}" 2>/dev/null || echo "(none)"
} >> "$GITHUB_STEP_SUMMARY"
