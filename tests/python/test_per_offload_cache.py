"""Per-offloaded-task compilation cache.

The LLVM codegen path (CPU / CUDA / AMDGPU) caches each offloaded task's compiled module in a per-Program in-memory
cache keyed on the task's IR (config + device caps + touched-SNode layout + re-id'd task body + autodiff mode). Editing
one offloaded task in a many-offload kernel makes the whole-kernel cache miss, but only the edited task recompiles: the
unchanged tasks are cloned out of the per-task cache. Asserted on counts (not wall time) exposed as
``kernel._primal.per_offload_cache_observations``.

``offline_cache=False`` so the on-disk whole-kernel cache never short-circuits codegen; the in-memory per-task cache is
what we exercise here. (A byte-identical kernel would instead hit the whole-kernel in-memory cache and never reach
per-task codegen, so it is not a useful probe of this layer.)
"""

import subprocess
import sys
import textwrap

import quadrants as qd

from tests import test_utils

# Distinct, deliberately-unusual constants so these tasks' cache keys do not collide with tasks other tests compiled.
# Each test needs its OWN set: the tiers under test are content-keyed and program-scoped, so two tests with the same
# kernel bodies would find the second one's "cold" phase already warm from the first.
_C = (61001.0, 61002.0, 61003.0, 61004.0)
_C_EDIT = 69999.0
_C2 = (62001.0, 62002.0, 62003.0, 62004.0)
_C2_EDIT = 68888.0
_C3 = (63001.0, 63002.0, 63003.0, 63004.0)
_C3_EDIT = 67777.0
_N = 8


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_offload_cache_one_construct_edit() -> None:
    # Four distinct top-level parallel loops => four independent offloaded tasks with four distinct cache keys.
    @qd.kernel
    def kernel_a(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]
        for i in range(_N):
            x[i] += _C[2]
        for i in range(_N):
            x[i] += _C[3]

    # Exactly one task changed (third loop's constant): only that task recompiles, the other three hit the cache.
    @qd.kernel
    def kernel_edit_one(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]
        for i in range(_N):
            x[i] += _C_EDIT
        for i in range(_N):
            x[i] += _C[3]

    arr = qd.ndarray(qd.f32, shape=(_N,))

    # Cold compile: the per-task cache starts empty, so every task is recompiled and stored.
    kernel_a(arr)
    obs_a = kernel_a._primal.per_offload_cache_observations
    assert obs_a.constructs_total == 4, obs_a
    assert obs_a.constructs_recompiled == 4, obs_a
    assert obs_a.constructs_cache_hit == 0, obs_a

    # Editing one offload recompiles exactly one task and reuses the other three from the per-task cache.
    kernel_edit_one(arr)
    obs_edit = kernel_edit_one._primal.per_offload_cache_observations
    assert obs_edit.constructs_total == 4, obs_edit
    assert obs_edit.constructs_recompiled == 1, obs_edit
    assert obs_edit.constructs_cache_hit == 3, obs_edit


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_cache_one_construct_edit() -> None:
    """Per-construct FRONTEND cache.

    Editing one construct recomputes exactly its frontend; the other three constructs are cloned out of the
    program-scoped construct cache.
    """

    @qd.kernel
    def kernel_a(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C2[0]
        for i in range(_N):
            x[i] += _C2[1]
        for i in range(_N):
            x[i] += _C2[2]
        for i in range(_N):
            x[i] += _C2[3]

    @qd.kernel
    def kernel_edit_one(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C2[0]
        for i in range(_N):
            x[i] += _C2[1]
        for i in range(_N):
            x[i] += _C2_EDIT
        for i in range(_N):
            x[i] += _C2[3]

    arr = qd.ndarray(qd.f32, shape=(_N,))

    kernel_a(arr)
    obs_a = kernel_a._primal.per_offload_cache_observations

    # Cold: empty construct cache, so all four constructs recompile their frontend.
    assert obs_a.frontend_constructs_total == 4, obs_a
    assert obs_a.frontend_constructs_recompiled == 4, obs_a
    assert obs_a.frontend_constructs_cache_hit == 0, obs_a

    # One edited construct recomputes its frontend; the three unchanged constructs are content-keyed cache hits
    # (shared across the two kernels, same ABI/config).
    kernel_edit_one(arr)
    obs_edit = kernel_edit_one._primal.per_offload_cache_observations
    assert obs_edit.frontend_constructs_total == 4, obs_edit
    assert obs_edit.frontend_constructs_recompiled == 1, obs_edit
    assert obs_edit.frontend_constructs_cache_hit == 3, obs_edit

    # Numerical correctness of the cache-hit clone path: on fresh arrays, both kernels must produce the exact sums.
    # A bad cloned construct would surface here (wrong / zeroed field).
    a2 = qd.ndarray(qd.f32, shape=(_N,))
    kernel_a(a2)
    assert all(abs(float(v) - sum(_C2)) < 1.0 for v in a2.to_numpy()), a2.to_numpy()
    e2 = qd.ndarray(qd.f32, shape=(_N,))
    kernel_edit_one(e2)
    expected = _C2[0] + _C2[1] + _C2_EDIT + _C2[3]
    assert all(abs(float(v) - expected) < 1.0 for v in e2.to_numpy()), e2.to_numpy()


# Body of the subprocess used by the cross-process test below. Each run is a fresh interpreter, so nothing can be
# reused in memory: any saved work has to come off disk. `variant` picks which constant the middle construct uses.
_XPROC = textwrap.dedent(
    """
    import sys
    import quadrants as qd

    cache_dir, variant = sys.argv[1], sys.argv[2]
    qd.init(arch=qd.cuda, offline_cache=True, offline_cache_file_path=cache_dir)

    N, C = {n}, {consts}
    middle = C[2] if variant == "base" else {edit}

    @qd.kernel
    def k(x: qd.types.ndarray()) -> None:
        for i in range(N):
            x[i] += C[0]
        for i in range(N):
            x[i] += C[1]
        for i in range(N):
            x[i] += middle
        for i in range(N):
            x[i] += C[3]

    arr = qd.ndarray(qd.f32, shape=(N,))
    k(arr)
    o = k._primal.per_offload_cache_observations
    expected = C[0] + C[1] + middle + C[3]
    ok = all(abs(float(v) - expected) < 1.0 for v in arr.to_numpy())
    print("RESULT", o.frontend_constructs_total, o.frontend_constructs_cache_hit,
          o.frontend_constructs_recompiled, ok)
    """
)


def _run_xproc(tmp_path, cache_dir, variant):
    # Must be a real file, not `python -c`: the AST transformer reads the kernel's source back off disk.
    script = tmp_path / "xproc_child.py"
    script.write_text(_XPROC.format(n=_N, consts=repr(_C3), edit=_C3_EDIT))
    proc = subprocess.run(
        [sys.executable, str(script), str(cache_dir), variant], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert line is not None, f"no RESULT line:\n{proc.stdout}\n{proc.stderr}"
    _, total, hit, recompiled, ok = line.split()
    return int(total), int(hit), int(recompiled), ok == "True"


@test_utils.test(arch=qd.cuda)
def test_cross_process_construct_reuse(tmp_path) -> None:
    """Constructs compiled by one process are reused by the next one.

    Both tiers are on disk here: the construct manifest names the tasks a construct produced, and the per-task
    artifact holds each task's cubin plus the launch metadata that makes the cubin usable. The second process edits
    one construct, so the whole-kernel `.qdc` key misses and those two tiers are the only thing that can save work --
    which is what distinguishes this from the in-memory tests above.
    """
    cache = tmp_path / "xproc_cache"

    total, hit, recompiled, ok = _run_xproc(tmp_path, cache, "base")
    assert (total, hit, recompiled) == (4, 0, 4), (total, hit, recompiled)
    assert ok

    # Fresh interpreter, one construct edited. The three unchanged constructs were never compiled in THIS process, so
    # a hit can only have come from the manifest + artifact tiers on disk.
    total, hit, recompiled, ok = _run_xproc(tmp_path, cache, "edit")
    assert (total, hit, recompiled) == (4, 3, 1), (total, hit, recompiled)
    assert ok, "edited kernel produced wrong values from reused artifacts"
