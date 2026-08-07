"""Tests for external (host) array kernel arguments - a raw `numpy.ndarray` or `torch.Tensor` passed where an ndarray is
expected, rather than a device-resident `qd.ndarray`.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

CONTAINERS = [
    "numpy",
    pytest.param("torch", marks=pytest.mark.needs_torch),
]


def _host_array(kind, n, fill):
    """Build an `n`-element int32 container of `kind` with every element preset to `fill`.

    `fill` must be non-zero for any test that checks untouched elements: the corruption being guarded against shows up
    as zeros, so a zero preset makes the broken and correct results identical.
    """
    if kind == "numpy":
        return np.full((n,), fill, dtype=np.int32)
    if kind == "torch":
        torch = pytest.importorskip("torch")
        # Deliberately a CPU tensor: one already resident on the compute device skips the staging path under test.
        return torch.full((n,), fill, dtype=torch.int32)
    raise AssertionError(f"unknown container kind {kind!r}")


def _as_numpy(arr, kind):
    if kind == "numpy":
        return arr
    return arr.cpu().numpy()


@test_utils.test()
@pytest.mark.parametrize("container", CONTAINERS)
def test_conditional_store_preserves_untouched_elements(container):
    """The reproducer from quadrants#841: only the even lanes are stored to, so the odd lanes must survive."""
    preset, stored = 7, 42

    @qd.kernel
    def store_even_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 0:
                out[pid] = stored

    out = _host_array(container, 8, preset)

    store_even_lanes(out)

    expected = np.array([stored, preset] * 4, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test()
@pytest.mark.parametrize("container", CONTAINERS)
def test_partial_range_store_preserves_tail(container):
    """The trigger is the access pattern, not the conditional: an unconditional store over a loop covering only the
    first half must leave the second half alone.

    The bound is passed as an argument rather than computed as `out.shape[0] // 2`, because an unlowered integer
    floordiv in a loop bound hits `QD_NOT_IMPLEMENTED` in the SPIR-V codegen - an unrelated gap that would mask this
    test with a compile error on Metal and Vulkan.
    """
    preset, stored = 7, 42

    @qd.kernel
    def store_prefix(out: qd.types.NDArray[qd.i32, 1], count: qd.i32):
        for pid in range(count):
            out[pid] = stored

    out = _host_array(container, 8, preset)

    store_prefix(out, 4)

    expected = np.array([stored] * 4 + [preset] * 4, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test()
@pytest.mark.parametrize("container", CONTAINERS)
def test_disjoint_writes_accumulate_across_launches(container):
    """Two launches, each writing a disjoint half, must compose. The staging buffer is allocated per launch and carries
    nothing from the previous one, so without the upload the second launch discards the first launch's output even
    though the two together cover every element.

    The two launches store distinct values so that a failure names the launch whose output was lost.
    """
    preset, first, second = 7, 42, 99

    @qd.kernel
    def store_even_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 0:
                out[pid] = first

    @qd.kernel
    def store_odd_lanes(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            if pid % 2 == 1:
                out[pid] = second

    out = _host_array(container, 8, preset)

    store_even_lanes(out)
    store_odd_lanes(out)

    expected = np.array([first, second] * 4, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test()
@pytest.mark.parametrize("container", CONTAINERS)
def test_read_modify_write_is_unaffected(container):
    """Control. A real load sets the mask's READ bit, so this path was always correct. It is here so that a failure of
    the tests above isolates the write-only elision rather than external-array staging in general."""
    preset = 7

    @qd.kernel
    def increment_every_element(out: qd.types.NDArray[qd.i32, 1]):
        for pid in range(out.shape[0]):
            out[pid] = out[pid] + 1

    out = _host_array(container, 8, preset)

    increment_every_element(out)

    expected = np.full((8,), preset + 1, dtype=np.int32)
    np.testing.assert_array_equal(_as_numpy(out, container), expected)


@test_utils.test()
@pytest.mark.parametrize("container", CONTAINERS)
def test_read_only_argument_is_not_clobbered(container):
    """Control. A read-only argument has the WRITE bit clear, so no readback happens and the caller's array must come
    back untouched."""
    preset = 7

    @qd.kernel
    def sum_into_first_element(src: qd.types.NDArray[qd.i32, 1], dst: qd.types.NDArray[qd.i32, 1]):
        for pid in range(src.shape[0]):
            dst[0] += src[pid]

    src = _host_array(container, 8, preset)
    dst = _host_array(container, 1, 0)

    sum_into_first_element(src, dst)

    np.testing.assert_array_equal(_as_numpy(src, container), np.full((8,), preset, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(dst, container), np.array([8 * preset], dtype=np.int32))
