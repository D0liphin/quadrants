"""Per-offloaded-task compilation cache.

The LLVM codegen path (CPU / CUDA / AMDGPU) caches each offloaded task's compiled module in a process-global cache
keyed on the task's IR (config + device caps + touched-SNode layout + re-id'd task body + autodiff mode). Editing one
offloaded task in a many-offload kernel must recompile only that task and reuse the rest. This is asserted on counts
(not wall time) exposed as ``kernel._primal.per_offload_cache_observations``.
"""

import quadrants as qd

from tests import test_utils

# Distinct, deliberately-unusual constants so these tasks' cache keys do not collide with tasks other tests compiled
# into the shared process-global cache.
_C = (61001.0, 61002.0, 61003.0, 61004.0)
_C_EDIT = 69999.0
_N = 8


@test_utils.test(arch=[qd.cpu, qd.cuda])
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

    # Byte-identical to kernel_a: after kernel_a compiles, every task here must hit the cache.
    @qd.kernel
    def kernel_same(x: qd.types.ndarray()) -> None:
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

    kernel_a(arr)
    obs_a = kernel_a._primal.per_offload_cache_observations
    assert obs_a.constructs_total == 4, obs_a
    assert obs_a.constructs_cache_hit + obs_a.constructs_recompiled == obs_a.constructs_total, obs_a

    # A byte-identical kernel reuses every task: full cache hit, zero recompiles.
    kernel_same(arr)
    obs_same = kernel_same._primal.per_offload_cache_observations
    assert obs_same.constructs_total == 4, obs_same
    assert obs_same.constructs_recompiled == 0, obs_same
    assert obs_same.constructs_cache_hit == 4, obs_same

    # Editing one offload recompiles exactly one task and reuses the other three.
    kernel_edit_one(arr)
    obs_edit = kernel_edit_one._primal.per_offload_cache_observations
    assert obs_edit.constructs_total == 4, obs_edit
    assert obs_edit.constructs_recompiled == 1, obs_edit
    assert obs_edit.constructs_cache_hit == 3, obs_edit
