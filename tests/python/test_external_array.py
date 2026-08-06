"""Tests for external (host) array kernel arguments - a raw `numpy.ndarray` or `torch.Tensor` passed where an ndarray is
expected, rather than a device-resident `qd.ndarray`.

On the gfx backends (Metal, Vulkan) such an argument is staged through a device buffer allocated per launch: a
host->device upload before the kernel, a device->host readback after. Both blits are gated on a per-argument access mask
built from the optimized IR (`irpass::detect_external_ptr_access_in_task`), which is where the round trip can go wrong.
The LLVM backends do not gate: CPU uses the host array in place, and CUDA stages it but uploads unconditionally.

`qd.ndarray` is parametrized alongside the host containers throughout. It never enters the staging path, so it pins the
reference behaviour the host containers must match.

## Partial writes must preserve untouched elements (quadrants#841)

The upload used to be gated on the mask's READ bit alone, so an argument that is only ever stored to got mask WRITE and
was never uploaded - the kernel ran against a freshly allocated device buffer, and the readback copied that whole
buffer, including the indices the kernel never wrote, back over the caller's array.

The trigger is the access pattern "stored to, never loaded from" rather than any particular syntax, so three shapes of it
are covered: a store under a conditional, a loop that does not span the whole array, and two launches that each write a
disjoint half. These tests assert on the *untouched* indices; asserting only the written elements passes even on a broken
build. Two controls (a read-modify-write, and a read-only argument) bracket them, so a failure isolates the write-only
elision rather than external-array staging in general.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

# The bug lives in the shared gfx runtime, so Metal and Vulkan are the interesting backends. CPU is included as a
# control: it uses the host array in place and has always been correct.
ARCHS = [qd.cpu, qd.metal, qd.vulkan]

# `quadrants` is the reference container: already device-resident, so it skips staging entirely and passes even on a
# broken build. Keeping it in the same parametrization makes the asymmetry explicit and guards against a "fix" that
# regresses the path which already worked. Only the torch case carries `needs_torch`; marking the whole test would
# wrongly gate the numpy and qd.ndarray cases behind an optional dependency.
CONTAINERS = [
    "numpy",
    pytest.param("torch", marks=pytest.mark.needs_torch),
    "quadrants",
]

FILL = 7
WRITTEN = 42


def _host_array(kind, n, fill=FILL):
    """Build an `n`-element int32 container of `kind` with every element preset to `fill`."""
    if kind == "numpy":
        return np.full((n,), fill, dtype=np.int32)
    if kind == "torch":
        torch = pytest.importorskip("torch")
        # Deliberately a CPU tensor: one already resident on the compute device skips the staging path under test.
        return torch.full((n,), fill, dtype=torch.int32)
    if kind == "quadrants":
        arr = qd.ndarray(qd.i32, (n,))
        arr.fill(fill)
        return arr
    raise AssertionError(f"unknown container kind {kind!r}")


def _as_numpy(arr, kind):
    if kind == "numpy":
        return arr
    if kind == "torch":
        return arr.cpu().numpy()
    return arr.to_numpy()


@test_utils.test(arch=ARCHS)
@pytest.mark.parametrize("container", CONTAINERS)
def test_conditional_store_preserves_untouched_elements(container):
    """The reproducer from quadrants#841: only the even lanes are stored to, so the odd lanes must survive."""

    @qd.kernel
    def store_even_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 0:
                out[pid] = WRITTEN

    out = _host_array(container, 8)

    store_even_lanes(out)

    expected = np.array([WRITTEN, FILL] * 4, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test(arch=ARCHS)
@pytest.mark.parametrize("container", CONTAINERS)
def test_partial_range_store_preserves_tail(container):
    """The trigger is the access pattern, not the conditional: an unconditional store over a loop covering only the
    first half must leave the second half alone.

    The bound is passed as an argument rather than computed as `out.shape[0] // 2`, because an unlowered integer
    floordiv in a loop bound hits `QD_NOT_IMPLEMENTED` in the SPIR-V codegen - an unrelated gap that would mask this
    test with a compile error on Metal and Vulkan.
    """

    @qd.kernel
    def store_prefix(out: qd.types.NDArray[qd.i32, 1], count: qd.i32):
        for pid in range(count):
            out[pid] = WRITTEN

    out = _host_array(container, 8)

    store_prefix(out, 4)

    expected = np.array([WRITTEN] * 4 + [FILL] * 4, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test(arch=ARCHS)
@pytest.mark.parametrize("container", CONTAINERS)
def test_disjoint_writes_accumulate_across_launches(container):
    """Two launches, each writing a disjoint half, must compose. The staging buffer is allocated per launch and carries
    nothing from the previous one, so without the upload the second launch discards the first launch's output even
    though the two together cover every element."""

    @qd.kernel
    def store_even_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 0:
                out[pid] = WRITTEN

    @qd.kernel
    def store_odd_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 1:
                out[pid] = WRITTEN

    out = _host_array(container, 8)

    store_even_lanes(out)
    store_odd_lanes(out)

    expected = np.full((8,), WRITTEN, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test(arch=ARCHS)
@pytest.mark.parametrize("container", CONTAINERS)
def test_read_modify_write_is_unaffected(container):
    """Control. A real load sets the mask's READ bit, so this path was always correct. It is here so that a failure of
    the tests above isolates the write-only elision rather than external-array staging in general."""

    @qd.kernel
    def increment_every_element(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            out[pid] = out[pid] + 1

    out = _host_array(container, 8)

    increment_every_element(out)

    expected = np.full((8,), FILL + 1, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test(arch=ARCHS)
@pytest.mark.parametrize("container", CONTAINERS)
def test_read_only_argument_is_not_clobbered(container):
    """Control. A read-only argument has the WRITE bit clear, so no readback happens and the caller's array must come
    back untouched."""

    @qd.kernel
    def sum_into_first_element(src: qd.types.NDArray[qd.i32, 1], dst: qd.types.NDArray[qd.i32, 1]):
        for pid in range(src.shape[0]):
            dst[0] += src[pid]

    src = _host_array(container, 8)
    dst = _host_array(container, 1, fill=0)

    sum_into_first_element(src, dst)

    np.testing.assert_array_equal(_as_numpy(src, container), np.full((8,), FILL, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(dst, container), np.array([8 * FILL], dtype=np.int32))
