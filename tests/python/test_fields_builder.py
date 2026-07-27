import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsRuntimeError

from tests import test_utils


@test_utils.test()
def test_fields_with_shape():
    shape = 5
    x = qd.field(qd.f32, shape=shape)

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i

    assign_field_single()
    for i in range(shape):
        assert x[i] == i

    y = qd.field(qd.f32, shape=shape)

    @qd.kernel
    def assign_field_multiple():
        for i in range(shape):
            y[i] = i * 2
        for i in range(shape):
            x[i] = i * 3

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 3
        assert y[i] == i * 2

    assign_field_single()
    for i in range(shape):
        assert x[i] == i


@test_utils.test()
def test_fields_builder_dense():
    shape = 5
    fb1 = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    fb1.dense(qd.i, shape).place(x)
    fb1.finalize()

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = qd.FieldsBuilder()
    y = qd.field(qd.f32)
    fb2.dense(qd.i, shape).place(y)
    z = qd.field(qd.f32)
    fb2.dense(qd.i, shape).place(z)
    fb2.finalize()

    @qd.kernel
    def assign_field_multiple():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


@test_utils.test(require=qd.extension.sparse)
def test_fields_builder_pointer():
    shape = 5
    fb1 = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    fb1.pointer(qd.i, shape).place(x)
    fb1.finalize()

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = qd.FieldsBuilder()
    y = qd.field(qd.f32)
    fb2.pointer(qd.i, shape).place(y)
    z = qd.field(qd.f32)
    fb2.pointer(qd.i, shape).place(z)
    fb2.finalize()

    @qd.kernel
    def assign_field_multiple_range_for():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple_range_for()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    @qd.kernel
    def assign_field_multiple_struct_for():
        for i in y:
            y[i] += 5
        for i in z:
            z[i] -= 5

    assign_field_multiple_struct_for()
    for i in range(shape):
        assert y[i] == i + 10
        assert z[i] == i + 5

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


# We currently only consider data types that all platforms support.
# See https://docs.taichi-lang.org/docs/type#primitive-types for more details.
@pytest.mark.parametrize("test_1d_size", [1, 10, 100])
@pytest.mark.parametrize("field_type", [qd.f32, qd.i32])
@test_utils.test()
def test_fields_builder_destroy(test_1d_size, field_type):
    def test_for_single_destroy_multi_fields():
        fb = qd.FieldsBuilder()
        for create_field_idx in range(10):
            field = qd.field(field_type)
            fb.dense(qd.i, test_1d_size).place(field)
        fb_snode_tree = fb.finalize()
        fb_snode_tree.destroy()

    def test_for_multi_destroy_multi_fields():
        fb0 = qd.FieldsBuilder()
        fb1 = qd.FieldsBuilder()

        for create_field_idx in range(10):
            field0 = qd.field(field_type)
            field1 = qd.field(field_type)

            fb0.dense(qd.i, test_1d_size).place(field0)
            fb1.pointer(qd.i, test_1d_size).place(field1)

        fb0_snode_tree = fb0.finalize()
        fb1_snode_tree = fb1.finalize()

        fb0_snode_tree.destroy()
        fb1_snode_tree.destroy()

    def test_for_raise_destroy_twice():
        fb = qd.FieldsBuilder()
        a = qd.field(qd.f32)
        fb.dense(qd.i, test_1d_size).place(a)
        c = fb.finalize()

        with pytest.raises(QuadrantsRuntimeError):
            c.destroy()
            c.destroy()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_snode_tree_count_limit_raises_not_corrupts():
    # The LLVM runtime keeps per-tree state (roots, root_mem_sizes) in fixed arrays of
    # kMaxNumSnodeTreesLlvm (512) entries indexed by tree id, written without bounds checking. A
    # 513th simultaneously-live tree used to overflow those arrays onto the adjacent thread_pool
    # pointer, so the next parallel kernel dereferenced garbage and crashed. Adding the tree past the
    # limit must now raise a clean error instead. Restricted to the LLVM backends: the gfx (Metal /
    # Vulkan) backends store roots in a growable vector and have no such limit.
    trees = []
    try:
        with pytest.raises(RuntimeError, match="maximum supported by the LLVM backend"):
            for _ in range(513):
                fb = qd.FieldsBuilder()
                x = qd.field(qd.f32)
                fb.dense(qd.i, 1).place(x)
                trees.append(fb.finalize())
    finally:
        for tree in trees:
            tree.destroy()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], require=qd.extension.sparse)
def test_snode_id_count_limit_raises_not_corrupts():
    # The LLVM runtime keeps per-snode state (element_lists, node_allocators, ambient_elements) in
    # fixed arrays of quadrants_max_num_snodes (2048) entries indexed by SNode id, written without
    # bounds checking. SNode ids come from a process-global monotonic counter that is reset only by a
    # full reset (qd.init), never by destroying trees, so reusing a runtime long enough overflows those
    # arrays and corrupts adjacent runtime state. Crossing the limit must raise a clean error instead.
    # Uses sparse (pointer) trees, which are the ones that perform the snode-id-indexed writes; trees
    # are destroyed each iteration (destroying does not rewind the snode counter).
    with pytest.raises(RuntimeError, match="maximum supported by the LLVM backend"):
        for _ in range(60):  # ~60 * 53 snodes per tree crosses 2048
            fb = qd.FieldsBuilder()
            fields = [qd.field(qd.f32) for _ in range(50)]
            block = fb.pointer(qd.i, 4).dense(qd.i, 4)
            for f in fields:
                block.place(f)
            tree = fb.finalize()
            tree.destroy()


@test_utils.test()
def test_free_all_memory():
    # free_all_memory() reclaims every field buffer (implicit-root fields and explicit
    # FieldsBuilder trees alike) and every ndarray buffer without a full qd.reset(): the
    # compiled kernels and compile config stay intact. Every field and ndarray created
    # before the call is invalid afterward.
    x = qd.field(qd.f32, shape=(64,))
    x.fill(1.0)

    fb = qd.FieldsBuilder()
    y = qd.field(qd.i32)
    fb.dense(qd.i, 32).place(y)
    fb.finalize()

    arr = qd.ndarray(qd.f32, shape=(8,))
    arr.fill(3.0)

    @qd.kernel
    def scale(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in a:
            a[i] = a[i] * 2.0

    scale(arr)
    num_compiled = impl.get_runtime().get_num_compiled_functions()
    assert num_compiled > 0

    qd.free_all_memory()

    # Compiled kernels are NOT cleared (unlike qd.reset()).
    assert impl.get_runtime().get_num_compiled_functions() == num_compiled
    # The ndarray handle is invalidated (its buffer was released).
    assert arr.arr is None

    # Runtime is still usable for freshly allocated fields and ndarrays, and the kernel
    # compiled before the call is reused as-is (no recompilation).
    z = qd.field(qd.f32, shape=(16,))
    z.fill(5.0)
    assert np.allclose(z.to_numpy(), 5.0)

    arr2 = qd.ndarray(qd.f32, shape=(8,))
    arr2.fill(3.0)
    scale(arr2)
    assert np.allclose(arr2.to_numpy(), 6.0)

    # Idempotent: a second call with no new allocations is a no-op.
    qd.free_all_memory()


@test_utils.test()
def test_field_initialize_zero():
    fb0 = qd.FieldsBuilder()
    a = qd.field(qd.i32)
    fb0.dense(qd.i, 1).place(a)
    c = fb0.finalize()
    a[0] = 5
    c.destroy()
    fb1 = qd.FieldsBuilder()
    b = qd.field(qd.i32)
    fb1.dense(qd.i, 1).place(b)
    d = fb1.finalize()
    assert b[0] == 0


@test_utils.test()
def test_field_builder_place_grad():
    @qd.kernel
    def mul(arr: qd.template(), out: qd.template()):
        for i in arr:
            out[i] = arr[i] * 2.0

    @qd.kernel
    def calc_loss(arr: qd.template(), loss: qd.template()):
        for i in arr:
            loss[None] += arr[i]

    arr = qd.field(qd.f32, needs_grad=True)
    fb0 = qd.FieldsBuilder()
    fb0.dense(qd.i, 10).place(arr, arr.grad)
    snode0 = fb0.finalize()
    out = qd.field(qd.f32)
    fb1 = qd.FieldsBuilder()
    fb1.dense(qd.i, 10).place(out, out.grad)
    snode1 = fb1.finalize()
    loss = qd.field(qd.f32)
    fb2 = qd.FieldsBuilder()
    fb2.place(loss, loss.grad)
    snode2 = fb2.finalize()
    arr.fill(1.0)
    mul(arr, out)
    calc_loss(out, loss)
    loss.grad[None] = 1.0
    calc_loss.grad(out, loss)
    mul.grad(arr, out)
    for i in range(10):
        assert arr.grad[i] == 2.0


@test_utils.test(arch=qd.cpu)
def test_fields_builder_numpy_dimension():
    shape = np.int32(5)
    fb = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    y = qd.field(qd.i32)
    fb.dense(qd.i, shape).place(x)
    fb.pointer(qd.j, shape).place(y)
    fb.finalize()
