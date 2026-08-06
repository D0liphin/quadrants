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

import pytest

import quadrants as qd

from tests import test_utils

# Distinct, deliberately-unusual constants so these tasks' cache keys do not collide with tasks other tests compiled.
_C = (61001.0, 61002.0, 61003.0, 61004.0)
_C_EDIT = 69999.0
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
    """Per-construct FRONTEND cache (S2 / §9.C).

    Only meaningful when the per-construct frontend split is enabled (QD_SPLIT_FRONTEND=1). The gate is a
    process-static, so this self-skips in a normal (split-off) run and asserts the construct-cache behavior when the
    suite is run with the split on. Editing one construct recomputes exactly its frontend; the other three constructs
    are cloned out of the program-scoped construct cache.
    """

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

    kernel_a(arr)
    obs_a = kernel_a._primal.per_offload_cache_observations
    if obs_a.frontend_constructs_total < 0:
        pytest.skip("per-construct frontend split not enabled (QD_SPLIT_FRONTEND unset)")

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
    assert all(abs(float(v) - sum(_C)) < 1.0 for v in a2.to_numpy()), a2.to_numpy()
    e2 = qd.ndarray(qd.f32, shape=(_N,))
    kernel_edit_one(e2)
    expected = _C[0] + _C[1] + _C_EDIT + _C[3]
    assert all(abs(float(v) - expected) < 1.0 for v in e2.to_numpy()), e2.to_numpy()
