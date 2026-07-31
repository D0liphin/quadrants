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

# Give MoltenVK a chance to log what it was doing right before it aborts.
# Level 1 = errors only (level 3 dumps ~126 supported extensions as noise on every init).
export MVK_CONFIG_LOG_LEVEL=1
# NB: do NOT set MTL_DEBUG_LAYER here. Production (4_test.sh / run_tests.py) does not enable
# Metal API Validation, and enabling it CHANGES the failure: the validation layer turns UB
# (e.g. a zero-grid dispatchThreadgroups from test_zero_outer_loop) into an immediate
# __assert_rtn/abort, so the crash lands at a different test/point than the real leg. We want
# the production abort, so leave the debug layer off and capture whatever MoltenVK does.

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

# macOS runners ship NO GNU `timeout`/`gtimeout`, so bounding a stage with `timeout N ...` fails with
# rc=127 (command not found) and the stage never runs. Portable shim: run "$@" in the background with a
# watchdog that TERM/KILLs it after N seconds. Returns the command's rc (143 if the watchdog TERMs it).
run_to() {
  local secs="$1"; shift
  "$@" &
  local cmd_pid=$!
  (
    sleep "$secs"
    kill -TERM "$cmd_pid" 2>/dev/null && { sleep 8; kill -KILL "$cmd_pid" 2>/dev/null; }
  ) &
  local wd_pid=$!
  local rc=0
  wait "$cmd_pid" 2>/dev/null || rc=$?
  kill "$wd_pid" 2>/dev/null           # command finished on its own -> stop the watchdog...
  pkill -P "$wd_pid" 2>/dev/null        # ...and its `sleep` child, so nothing lingers.
  wait "$wd_pid" 2>/dev/null || true
  return "$rc"
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
    '"GUARDCHK to=%s eqf64=%s idn=%s idf=%s arch=%s eq_vulkan=%s plat=%s" % '
    '(getattr(dtype, "to_string", lambda: str(dtype))(), dtype == qd.f64, id(dtype), id(qd.f64), '
    'arch, arch == qd.vulkan, _os.uname().sysname) + chr(10))\n'
)
open(p, "w").write(s.replace(anchor, anchor + log, 1))
print("patched guardlog into", p)
PY
python "${RUNNER_TEMP}/guard_instr.py"
python -c "import ast; ast.parse(open('tests/python/test_simt.py').read()); print('test_simt.py parses OK')"
grep -n -A2 "arch = qd.lang.impl.current_cfg().arch" tests/python/test_simt.py | head

# --- phase log ------------------------------------------------------------------------
# The guard only runs in a test's body (call phase), so it cannot see an abort that happens
# in per-test setup/teardown (the @test_utils.test(arch=qd.gpu) qd.init/reset + MoltenVK
# GPU-context churn). Append setup/call/teardown hooks so the LAST line before the abort is
# the exact test + phase where MoltenVK died -- and preceding lines show any failing test
# (the suite kills+restarts an xdist worker on failure to reset GPU state; at -t 1 there is
# no worker to recycle, so state accumulates until the driver aborts).
export PHASELOG="${CRASH_OUT}/phaselog.txt"
export RSSLOG="${CRASH_OUT}/rsslog.txt"
: > "${PHASELOG}"
: > "${RSSLOG}"
cat >> tests/python/conftest.py <<'PY'


# --- DIAG (vulkan_simt_crash.sh): per-phase logger to locate the MoltenVK abort ---
def _qd_phaselog(when, nodeid):
    try:
        with open(os.environ.get("PHASELOG", "/tmp/phaselog.txt"), "a") as _f:
            _f.write(when + " " + str(nodeid) + "\n")
    except Exception:
        pass


# --- DIAG (investigation 2): per-test resident-set size, to see whether process-wide
# Metal/MoltenVK/host state grows monotonically across qd.init()/qd.reset() cycles (leak-like)
# or stays flat (pure driver-handle exhaustion). Uses ru_maxrss (peak, monotonic) plus a live
# RSS read via `ps` so we can distinguish a rising-then-plateau curve from unbounded growth.
import resource as _qd_resource

_qd_test_counter = [0]


def _qd_rsslog(nodeid):
    _qd_test_counter[0] += 1
    n = _qd_test_counter[0]
    # Fork-free (no per-test `ps` subprocess -- that both perturbs the run and gets more expensive as
    # RSS grows, confounding the very measurement). ru_maxrss is bytes on macOS, kB on Linux; a rising
    # peak across tests = accumulation, a plateau = bounded.
    maxrss = _qd_resource.getrusage(_qd_resource.RUSAGE_SELF).ru_maxrss
    maxrss_kb = maxrss // 1024 if maxrss > 10_000_000 else maxrss
    try:
        with open(os.environ.get("RSSLOG", "/tmp/rsslog.txt"), "a") as _f:
            _f.write("%d max_rss_kb=%d %s\n" % (n, maxrss_kb, str(nodeid)))
    except Exception:
        pass


def pytest_runtest_setup(item):
    _qd_phaselog("SETUP", item.nodeid)


def pytest_runtest_call(item):
    _qd_phaselog("CALL", item.nodeid)


def pytest_runtest_teardown(item, nextitem):
    _qd_phaselog("TEARDOWN", item.nodeid)
    _qd_rsslog(item.nodeid)
PY
python -c "import ast; ast.parse(open('tests/python/conftest.py').read()); print('conftest.py parses OK')"

# We let the process abort naturally if fix(1) is absent; macOS ReportCrash writes an .ips (with the
# faulting-thread backtrace) that collect_and_dump parses. This wheel is BUILT FROM THIS BRANCH, so
# fix(1) (VulkanStream::submit throws instead of RHI_ASSERT/abort) is compiled in. The f64-guard
# stages are gone (already answered: f64 is a red herring; the abort is submit() during teardown).
#
# Every heavy stage is wrapped in `timeout` so that even on a slow/"bad-VM-lottery" runner the job
# never hits the hard job-level timeout (which would SKIP the always() upload step and lose all
# artifacts). Order matters: the FAST, VM-robust validations run first, the slow authentic repro last.

# --- Stage A (fast, ~minutes): controlled saturation probe -----------------------------
# Directly drives the accumulation the real suite triggers only after ~1000 tests: N cycles of
# init(vulkan) -> compile+launch a fresh kernel (new pipeline each cycle) -> readback (submit+sync)
# -> reset(). Answers BOTH questions quickly and VM-independently:
#   fix(1): if submit fails it should raise RuntimeError (rc=1, "THREW"), NOT Abort trap:6 (rc=134,.ips).
#   leak(2): ru_maxrss per cycle -> monotonic climb (host accumulation) vs plateau (driver-handle only).
SATPROBE="${RUNNER_TEMP}/saturate_probe.py"
cat > "${SATPROBE}" <<'PY'
import os, resource
import quadrants as qd

RSSLOG = os.environ.get("SAT_RSSLOG", "/tmp/sat_rss.txt")
open(RSSLOG, "w").close()
N = int(os.environ.get("SAT_CYCLES", "6000"))
print("SATURATE probe: up to %d init/reset+submit cycles on vulkan" % N, flush=True)


def _rss_kb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 1024 if r > 10_000_000 else r


i = 0
try:
    for i in range(N):
        qd.init(arch=qd.vulkan)
        m = 256 + (i % 128)
        fld = qd.field(dtype=qd.f32, shape=m)

        @qd.kernel
        def kern(s: qd.f32):
            for j in range(m):
                fld[j] = s * j

        kern(1.0 + (i % 5))
        _ = fld.to_numpy()  # force submit + sync
        qd.reset()
        if i % 25 == 0:
            line = "cycle=%d max_rss_kb=%d" % (i, _rss_kb())
            print("SAT", line, flush=True)
            with open(RSSLOG, "a") as fh:
                fh.write(line + "\n")
    print("SATURATE_NO_REPRO completed %d cycles final_max_rss_kb=%d" % (N, _rss_kb()), flush=True)
except Exception as e:
    print("SATURATE_FIX1_THREW cycle=%d type=%s msg=%s" % (i, type(e).__name__, str(e)[:200]), flush=True)
    with open(RSSLOG, "a") as fh:
        fh.write("THREW cycle=%d %s: %s\n" % (i, type(e).__name__, str(e)[:200]))
    raise
PY
run_stage saturate_probe run_to 1200 env SAT_RSSLOG="${CRASH_OUT}/sat_rss.txt" SAT_CYCLES=6000 python "${SATPROBE}"

# --- Stage B (the production FIX): full serial vulkan suite WITH proactive worker recycle ----
# QD_WORKER_RECYCLE_EVERY makes run_tests.py force a single xdist worker even at -t 1, and conftest
# recycles it every N completed tests, bounding accumulation. Even on a bad VM this should stay fast
# and reach 100% (rc=0). This is the change that gets the vulkan leg to >=99%.
run_stage recycle_full \
  run_to 4200 env QD_WORKER_RECYCLE_EVERY=25 \
  python tests/run_tests.py -v -r 1 --arch vulkan -m "not needs_torch" -t 1

# --- Stage C (authentic fix(1) + leak curve, slow, LAST): full serial vulkan, NO recycle -----
# The exact failing-leg command. Pre-fix this hard-aborts (rc=134,.ips) ~63% in. With fix(1) it must
# instead raise a catchable RuntimeError (rc!=134, no .ips) and keep going. Also produces the full
# RSS-vs-test curve for investigation (2). Bounded by timeout so artifacts always upload.
run_stage norecycle_full \
  run_to 4200 \
  python tests/run_tests.py -v -r 1 --arch vulkan -m "not needs_torch" -t 1

# --- job summary -----------------------------------------------------------------------
{
  echo "## Vulkan SIMT crash: fix(1) validation + accumulation/leak probe"
  echo ""
  echo "fix(1): VulkanStream::submit() throws QuadrantsRuntimeError on vkQueueSubmit failure instead of"
  echo "RHI_ASSERT->assert->abort(). PASS = no .ips anywhere; saturate_probe/norecycle_full show a"
  echo "RuntimeError (rc=1) instead of Abort trap:6 (rc=134); recycle_full reaches 100% (rc=0)."
  echo ""
  echo '```'
  cat "${STAGE_RESULTS}"
  echo '```'
  echo ""
  echo "### Stage A saturate_probe: RSS per init/reset+submit cycle (leak vs plateau) + fix(1) outcome"
  echo '```'
  if [ -s "${CRASH_OUT}/sat_rss.txt" ]; then
    awk 'NR==1 || NR%10==0' "${CRASH_OUT}/sat_rss.txt"
    echo "-- last 5 --"; tail -n 5 "${CRASH_OUT}/sat_rss.txt"
  else
    echo "(no sat_rss)"
  fi
  echo '```'
  echo ""
  echo "### Phase log tail (final line = test+phase where the run stopped)"
  echo '```'
  tail -n 30 "${PHASELOG}" 2>/dev/null || echo "(no phaselog)"
  echo '```'
  echo ""
  echo "### Investigation (2): full-suite RSS vs test index (norecycle_full)"
  echo '```'
  if [ -s "${RSSLOG}" ]; then
    total=$(wc -l < "${RSSLOG}")
    echo "total tests logged: ${total}"
    echo "-- sampled every ~100 tests --"
    awk 'NR==1 || NR%100==0' "${RSSLOG}"
    echo "-- last 5 --"
    tail -n 5 "${RSSLOG}"
  else
    echo "(no rsslog)"
  fi
  echo '```'
  echo ""
  echo "### Crash reports collected (should be EMPTY if fix(1) works - no more abort)"
  ls -1 "${CRASH_OUT}"/*.ips 2>/dev/null || echo "(no .ips - good, means no abort)"
} >> "$GITHUB_STEP_SUMMARY"
