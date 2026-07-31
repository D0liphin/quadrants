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
# Level 1 = errors only (level 3 dumps ~126 supported extensions as noise on every init).
export MVK_CONFIG_LOG_LEVEL=1
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
  # ReportCrash can lag under heavy post-abort system load; show what actually landed.
  echo "--- DiagnosticReports contents (${tag}) ---"
  ls -la "${DR_USER}" 2>/dev/null || true
  sudo -n ls -la "${DR_SYS}" 2>/dev/null || ls -la "${DR_SYS}" 2>/dev/null || true
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
  # SIGABRT surfaces as rc 134; ReportCrash can take a while to flush the .ips under load.
  sleep 25
  collect_and_dump_crashes "${label}"
  echo "STAGE_RESULT ${label} rc=${rc}" >> "${STAGE_RESULTS}"
}

pip install --prefer-binary --group test
export QD_LIB_DIR="$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
echo "python=$(python -V 2>&1) quadrants=$(python -c 'import quadrants as qd; print(getattr(qd,"__version__","?"))' 2>&1)"

# --- instrument the f64 skip guard (root-cause probe) ----------------------------------
# Confirmed interactively: in the full vulkan suite the abort hits at
# test_subgroup_inclusive_mul_tiled[f64] and NO "SKIPPED" is printed, while the sibling
# test_subgroup_inclusive_add_tiled[f64] SKIPS correctly seconds earlier. So the guard
# _skip_if_f64_unsupported() (which pytest.skips f64 when current_cfg().arch == qd.vulkan on
# Darwin) stops firing after enough accumulation -- current_cfg().arch drifts away from
# qd.vulkan. Log what arch the guard actually sees on every call into an uploaded file so we
# can read the value at the crashing test (the last GUARDCHK line before the abort). This is
# a pure logging line inserted after `arch = ...`; it does not change control flow.
export GUARDLOG="${CRASH_OUT}/guardlog.txt"
: > "${GUARDLOG}"
# Inject a pure logging line right after `arch = current_cfg().arch`. Use a here-doc'd
# python patcher (not perl -pi, whose s/// replacement is double-quote-interpolated and turns
# a "\n" inside the string into a real newline -> unterminated string literal) and build the
# record's newline with chr(10) so there is no backslash-escape to mangle.
cat > "${RUNNER_TEMP}/guard_instr.py" <<'PY'
p = "tests/python/test_simt.py"
s = open(p).read()
anchor = "    arch = qd.lang.impl.current_cfg().arch\n"
assert s.count(anchor) == 1, ("anchor count", s.count(anchor))
log = (
    '    import os as _os; open(_os.environ.get("GUARDLOG", "/tmp/guardlog.txt"), "a").write('
    '"GUARDCHK dtype=%r arch=%r eq_vulkan=%r eq_metal=%r plat=%r" % '
    '(dtype, arch, arch == qd.vulkan, arch == qd.metal, _os.uname().sysname) + chr(10))\n'
)
open(p, "w").write(s.replace(anchor, anchor + log, 1))
print("patched guardlog into", p)
PY
python "${RUNNER_TEMP}/guard_instr.py"
python -c "import ast; ast.parse(open('tests/python/test_simt.py').read()); print('test_simt.py parses OK')"
grep -n -A2 "arch = qd.lang.impl.current_cfg().arch" tests/python/test_simt.py | head

# NB: we run each repro DIRECTLY (not under lldb). lldb --batch on these macOS runners
# stops at the initial exec and quits without ever running the program (stop reason=exec),
# so it produced no data. Instead we let the process abort naturally; macOS ReportCrash
# writes an .ips crash report (with the faulting-thread backtrace) that collect_and_dump
# parses. This wheel is BUILT FROM main (the published 1.2.0 gives a clean "Type f64 not
# supported" error and does not reproduce the abort).

# --- Stage 1: does plain (non-subgroup) f64 field I/O abort on vulkan? -----------------
# Tests the broad claim in the skip message ("MoltenVK does not support f64"). If this
# aborts, f64 is fundamentally unusable on vulkan/Darwin; if it raises a clean Python
# error, f64 fails gracefully and only some other path aborts.
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

# --- Stage 2: the exact crashing case, in isolation, vulkan-only serial ----------------
# Establishes the baseline: with a vulkan-only run the _skip_if_f64_unsupported guard
# fires and this SKIPS (no crash). (Confirmed: 1 skipped.)
run_stage isolate_f64_mul_vulkan_only \
  python tests/run_tests.py test_simt -k "test_subgroup_inclusive_mul_tiled and dtype3" --arch vulkan -t 1 -v -s

# --- Stage 3: neuter the skip guard, run f64 mul_tiled vulkan-only ---------------------
# Decisive: does the f64 vulkan subgroup prefix-product kernel ABORT the driver by itself
# (real backend bug, guard is load-bearing), or raise a clean "Type f64 not supported"
# RuntimeError (=> the CI abort needs accumulation / some other trigger)?
cp tests/python/test_simt.py "${RUNNER_TEMP}/test_simt.orig.py"
perl -0pi -e 's/(def _skip_if_f64_unsupported\(dtype\):\n)/$1    return  # DIAG: neutered to test the raw f64 vulkan subgroup path\n/' tests/python/test_simt.py
grep -n -A2 "def _skip_if_f64_unsupported" tests/python/test_simt.py | head
run_stage f64_mul_noskip \
  python tests/run_tests.py test_simt -k "test_subgroup_inclusive_mul_tiled and dtype3" --arch vulkan -t 1 -v -s
cp "${RUNNER_TEMP}/test_simt.orig.py" tests/python/test_simt.py   # restore

# --- Stage 4a: vulkan-only test_simt.py, serial ---------------------------------------
# The failing production leg is VULKAN-ONLY (MAC_TEST_ARCH=vulkan) with a single serial
# worker; it aborts at ~63% in test_subgroup_inclusive_mul_tiled[f64] even though that
# same test SKIPS in isolation (stage 2). Try to reproduce with just test_simt.py first.
run_stage repro_test_simt_vulkan \
  python tests/run_tests.py test_simt --arch vulkan -t 1 -v

# --- Stage 4b: full vulkan-only "not needs_torch" repro (only if 4a did not crash) -----
# The exact failing-leg command. Bounded: the production legs aborted ~18 min in.
if ls "${CRASH_OUT}"/repro_test_simt_vulkan__* >/dev/null 2>&1; then
  echo "test_simt-only already reproduced the crash; skipping full vulkan repro"
  echo "STAGE_RESULT repro_full_vulkan skipped" >> "${STAGE_RESULTS}"
else
  run_stage repro_full_vulkan \
    python tests/run_tests.py -v -r 1 --arch vulkan -m "not needs_torch" -t 1
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
  echo "### Guard (_skip_if_f64_unsupported) arch drift"
  echo "eq_vulkan=True calls: $(grep -c 'eq_vulkan=True'  "${GUARDLOG}" 2>/dev/null || echo 0); "\
       "eq_vulkan=False calls: $(grep -c 'eq_vulkan=False' "${GUARDLOG}" 2>/dev/null || echo 0)"
  echo ""
  echo "Last 25 guard calls (the final line is the crashing test_subgroup_inclusive_mul_tiled[f64]):"
  echo '```'
  tail -n 25 "${GUARDLOG}" 2>/dev/null || echo "(no guardlog)"
  echo '```'
  echo ""
  echo "### Crash reports collected"
  ls -1 "${CRASH_OUT}" 2>/dev/null || echo "(none)"
} >> "$GITHUB_STEP_SUMMARY"
