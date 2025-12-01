#!/usr/bin/env python3

import time

import jax
import numpy as np
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec

AXIS_NAME = "i"
N_DEVICES = 4

# Toggle this if you want to use `jax.debug.print`.
#
# Setting `ENABLE_INTERPRET=True` will also enable the following effects:
# * Dynamic out-of-bounds access detection
# * Dynamic race condition detection
# * Uninitialized data is initially filled with NaN values
#
ENABLE_DEBUG = False

N_BATCH = 256
K1 = 1024
K2 = 4096 


################################################################################
# Pallas RDMA helpers for convenience (already written)


def pallas_get_my_device_id():
    """
    Returns the index of the current device within the 4-device ring (0, 1, 2, or 3).

    This function returns a *traced* JAX/Pallas integer, not a Python `int`; at trace time, its
    concrete value is not known.

    You can perform basic arithmetic operations on the returned value (`+`, `-`, `*`, `//`, `%`,
    etc.) to compute other traced integers, and you can use it in Pallas APIs like
    `pallas_rdma_start`, or Pallas array indexing operations like `arr[...]` or `arr.at[...]`.

    You *cannot* use the value anywhere a concrete Python `int` is required, such as to index into
    a Python list, or in control flow conditions like `if pallas_get_my_device_id() == 2: ...`.
    """

    return lax.axis_index(AXIS_NAME)


def pallas_rdma_start(*, src_ref, dst_ref, dst_device_id, src_send_sem, dst_recv_sem):
    """
    During Pallas tracing, emits a Remote Direct Memory Access (RDMA) operation that asynchronously
    copies from the current device's VMEM to another device's VMEM.

    You can launch multiple asynchronous RDMA transfers back-to-back, and don't need to wait for
    each to complete before starting the next one. The RDMA transfers will be queued up in the ICI
    DMA engine on a per-link basis, and will execute in the order in which they were launched.

    (Note that, like all traced operations, this function does not actually perform the RDMA copy
    when it is called from Python; instead, it emits instructions into the Pallas kernel being
    traced, which will be executed later on the TPU after tracing is complete.)

    Arguments:

    * `src_ref`: A Pallas array reference pointing to the source data for the transfer, in the
      current device's address space.

    * `dst_ref`: A Pallas array reference pointing to the destination for the transfer, in the
      *remote* device's address space.

    * `dst_device_id`: A traced JAX/Pallas integer specifying the index in the 4-device ring of the
      remote device to which data will be sent (0, 1, 2, or 3). Should usually be a direct neighbor
      of the current device (computed via doing some math on `pallas_get_my_device_id()`).

    * `src_send_sem`: A Pallas DMA semaphore reference in the current device's address space. The
      ICI DMA engine will signal this semaphore on the current device when it has finished reading
      all data from `src_ref`.

    * `dst_recv_sem`: A Pallas DMA semaphore reference in the *remote* device's address space. The
      ICI DMA engine on the remote device will signal this semaphore when it has finished writing
      all data to `dst_ref`.

    Important constraints:

    * After starting an RDMA transfer, you *must* subsequently call `pallas_rdma_wait_send` on
     the sender and `pallas_rdma_wait_recv` on the receiver with the same semaphore references
     before the kernel completes.

    * A DMA semaphore can only participate in *one RDMA operation at a time*. You must not start
      another RDMA operation using the same `src_send_sem` or `dst_recv_sem` until you have waited
      on the previous operation using `pallas_rdma_wait_send` or `pallas_rdma_wait_recv`.
    """

    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=dst_ref,
        send_sem=src_send_sem,
        recv_sem=dst_recv_sem,
        device_id=dst_device_id,
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).start()


def pallas_rdma_wait_send(*, src_ref, src_send_sem):
    """
    During Pallas tracing, emits a wait for the "send" half of a previously-started RMDA operation
    originating from the current device.

    (Note that, like all traced opertaions, this function does not actually perform the wait when it
    is called from Python; instead, it emits instructions into the Pallas kernel being traced, which
    will be executed later on the TPU after tracing is complete.)

    Arguments:

    * `src_ref`: A Pallas array reference in the current device's address space that was used as the
      source for a previously-started RDMA transfer.

    * `src_send_sem`: The Pallas DMA semaphore in the current device's address space which was used
      as the "send" semaphore for the RDMA transfer.
    """

    pltpu.make_async_remote_copy(
        src_ref=src_ref,
        dst_ref=src_ref,  # ignored by 'wait_send'
        send_sem=src_send_sem,
        recv_sem=src_send_sem,  # ignored by 'wait_send'
        device_id=0,  # ignored by 'wait_send'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_send()


def pallas_rdma_wait_recv(*, dst_ref, dst_recv_sem):
    """
    During Pallas tracing, emits a wait for the "receive" half of a previously-started RDMA
    operation targeting the current device.

    (Note that, like all traced opertaions, this function does not actually perform the wait when it
    is called from Python; instead, it emits instructions into the Pallas kernel being traced,
    which will be executed later on the TPU after tracing is complete.)

    Arguments:

    * `dst_ref`: A Pallas array reference in the current device's address space that was used as the
      destination for a previously-started RDMA transfer. (started on a remote device)

    * `dst_recv_sem`: The Pallas DMA semaphore in the current device's address space which was used
      as the "recv" semaphore for the RDMA transfer.
    """

    pltpu.make_async_remote_copy(
        src_ref=dst_ref,  # ignored by 'wait_recv'
        dst_ref=dst_ref,
        send_sem=dst_recv_sem,  # ignored by 'wait_recv'
        recv_sem=dst_recv_sem,
        device_id=0,  # ignored by 'wait_recv'
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    ).wait_recv()


################################################################################
# Part 2: Overlapping computation and communication

## <--- your code here --->


def matmul_pallas_scratch_specs(x, w):
    return {
        # TODO: your code here
    }


def matmul_pallas_kernel(x_ref, w_ref, out_ref, scratch_refs):
    """
    Computes the matrix multiplication x_ref @ w_ref

    This is called in two shape configurations:
    * x_ref: [N_BATCH, K1], w_ref: [K1, K2] --> out_ref: [N_BATCH, K2]
    * x_ref: [N_BATCH, K2], w_ref: [K2, K1] --> out_ref: [N_BATCH, K1]

    All arrays are in VMEM and have dtype bfloat16.
    """

    # TODO: your code here
    pass


def all_gather_matmul_pallas_scratch_specs(x):
    return {
        # TODO: your code here
    }


def all_gather_matmul_pallas_kernel(x_ref, w1_ref, out_ref, scratch_refs):
    """
    Computes all_gather(x_ref) @ w1_ref, where the gather occurs along the second dimension.

    Shapes:
    * x_ref: [N_BATCH, K1 / N_DEVICES]
    * w1_ref: [K1, K2]
    * out_ref: [N_BATCH, K2]

    All arrays are in VMEM and have dtype bfloat16.
    """

    # TODO: your code here
    pass


def matmul_reduce_scatter_pallas_scratch_specs(x):
    return {
        # TODO: your code here
    }


def matmul_reduce_scatter_pallas_kernel(x_ref, w2_ref, out_ref, scratch_refs):
    """
    Computes reduce_scatter(x_ref @ w2_ref), sharded along the second dimension.

    Shapes:
    * x_ref: [N_BATCH, K2]
    * w2_ref: [K2, K1]
    * out_ref: [N_BATCH, K1 / N_DEVICES]
    """

    # TODO: your code here
    pass


def neural_network_pallas_scratch_specs(x, w1_refs, w2_refs):
    return {
        # TODO: your code here
    }


def neural_network_pallas_kernel(init_x_ref, w1_refs, w2_refs, out_ref, scratch_refs):
    """
    Computes the output of the neural network defined by the following pseudocode:

    x = init_x
    for w1, w2 in zip(w1_refs, w2_refs):
        x = x @ w1
        x = gelu(x)
        x = x @ w2
        x = all_reduce(x)
    out = x

    Arguments:
    * init_x_ref: Pallas array reference with shape [N_BATCH, K1]. Read-only.
    * w1_refs: List of Pallas array refs each with shape [K1, K2]. Read-only.
    * w2_refs: List of Pallas array refs each with shape [K2, K1]. Read-only.
    * out_ref: Pallas array reference with shape [N_BATCH, K1]. Should be written to.
    """

    # TODO: your code here
    pass


## <--- /your code here --->


################################################################################
##                YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.              ##
################################################################################


def matmul_reference(x, w):
    return x @ w


def all_gather_matmul_reference(x, w):
    x = lax.all_gather(x, axis_name=AXIS_NAME, axis=1, tiled=True)
    return x @ w


def matmul_reduce_scatter_reference(x, w):
    x = x @ w
    return lax.psum_scatter(x, axis_name=AXIS_NAME, scatter_dimension=1, tiled=True)


def neural_network_reference(x, w1_list, w2_list):
    for w1, w2 in zip(w1_list, w2_list):
        x = x @ w1
        x = jax.nn.gelu(x)
        x = x @ w2
        x = lax.psum(x, axis_name=AXIS_NAME)
    return x


def get_interpret_params():
    if ENABLE_DEBUG:
        return pltpu.InterpretParams(detect_races=True)
    else:
        return None


def launch_kernel_looped(
    *,
    kernel,
    out_shape,
    scratch_shapes,
    loop_iters,
    input_arrays,
    in_specs,
    calibrate_only=False,
):
    def kernel_wrapper(input_refs, loop_iters_ref, out_ref, loop_sems, scratch_refs):
        my_id = pallas_get_my_device_id()
        other_devices = [
            lax.rem(my_id + offset, N_DEVICES) for offset in range(1, N_DEVICES)
        ]

        def loop_body(_, __):
            out_ref[...] = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)
            pl.delay(1)  # optimization barrier

            if not calibrate_only:
                kernel(*input_refs, out_ref, scratch_refs)

            # Double-semaphore handshake to avoid runahead in cases where 'kernel' itself does no
            # synchronization.
            for loop_sem_idx in range(2):
                for other_device in other_devices:
                    pltpu.semaphore_signal(
                        loop_sems.at[loop_sem_idx],
                        device_id=other_device,
                        device_id_type=pltpu.DeviceIdType.LOGICAL,
                    )
                pltpu.semaphore_wait(loop_sems.at[loop_sem_idx], len(other_devices))

        lax.fori_loop(0, loop_iters_ref[0], loop_body, None)

    # Accept loop_iters as either a Python int or a JAX scalar array
    if isinstance(loop_iters, int):
        loop_iters_arr = jnp.array([loop_iters], dtype=jnp.int32)
    else:
        loop_iters_arr = loop_iters.reshape((1,))

    return pl.pallas_call(
        kernel=kernel_wrapper,
        out_shape=out_shape,
        scratch_shapes=(pltpu.SemaphoreType.REGULAR(shape=(2,)), scratch_shapes),
        in_specs=(
            in_specs,
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20),
        interpret=get_interpret_params(),
    )(input_arrays, loop_iters_arr)


def matmul_pallas_launch(x, w, *, loop_iters, calibrate_only=False):
    """Launch the simple matmul Pallas kernel."""
    assert x.ndim == 2 and w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.dtype == jnp.bfloat16 and w.dtype == jnp.bfloat16

    out_shape = jax.ShapeDtypeStruct(shape=(x.shape[0], w.shape[1]), dtype=x.dtype)

    return launch_kernel_looped(
        kernel=matmul_pallas_kernel,
        out_shape=out_shape,
        scratch_shapes=matmul_pallas_scratch_specs(x, w),
        loop_iters=loop_iters,
        input_arrays=(x, w),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        calibrate_only=calibrate_only,
    )


def all_gather_matmul_pallas_launch(x, w1, *, loop_iters, calibrate_only=False):
    """Launch the all_gather_matmul Pallas kernel."""
    assert x.ndim == 2 and w1.ndim == 2
    assert x.shape[1] * N_DEVICES == w1.shape[0]  # x is sharded on K1, w1 has full K1
    assert x.dtype == jnp.bfloat16 and w1.dtype == jnp.bfloat16

    # Output has same K2 sharding as w1
    out_shape = jax.ShapeDtypeStruct(shape=(x.shape[0], w1.shape[1]), dtype=x.dtype)

    return launch_kernel_looped(
        kernel=all_gather_matmul_pallas_kernel,
        out_shape=out_shape,
        scratch_shapes=all_gather_matmul_pallas_scratch_specs(x),
        loop_iters=loop_iters,
        input_arrays=(x, w1),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        calibrate_only=calibrate_only,
    )


def matmul_reduce_scatter_pallas_launch(x, w2, *, loop_iters, calibrate_only=False):
    """Launch the matmul_reduce_scatter Pallas kernel."""
    assert x.ndim == 2 and w2.ndim == 2
    assert x.shape[1] == w2.shape[0]  # x and w2 both sharded on K2
    assert x.dtype == jnp.bfloat16 and w2.dtype == jnp.bfloat16

    # Output K1 is sharded across devices after reduce-scatter
    out_shape = jax.ShapeDtypeStruct(
        shape=(x.shape[0], w2.shape[1] // N_DEVICES), dtype=x.dtype
    )

    return launch_kernel_looped(
        kernel=matmul_reduce_scatter_pallas_kernel,
        out_shape=out_shape,
        scratch_shapes=matmul_reduce_scatter_pallas_scratch_specs(x),
        loop_iters=loop_iters,
        input_arrays=(x, w2),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        calibrate_only=calibrate_only,
    )


def neural_network_pallas_launch(
    x, w1_list, w2_list, *, loop_iters, calibrate_only=False
):
    """Launch the neural_network Pallas kernel."""
    assert x.ndim == 2
    assert all(w.ndim == 2 for w in w1_list)
    assert all(w.ndim == 2 for w in w2_list)
    assert len(w1_list) == len(w2_list)
    assert x.dtype == jnp.bfloat16

    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)

    # Stack weights for passing to kernel
    w1_stacked = jnp.stack(w1_list, axis=0)
    w2_stacked = jnp.stack(w2_list, axis=0)

    return launch_kernel_looped(
        kernel=neural_network_pallas_kernel,
        out_shape=out_shape,
        scratch_shapes=neural_network_pallas_scratch_specs(x, w1_list, w2_list),
        loop_iters=loop_iters,
        input_arrays=(x, w1_stacked, w2_stacked),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        calibrate_only=calibrate_only,
    )


def make_activation(rng, shape):
    """Create a bf16 activation tensor with normal initialization."""
    return jax.random.normal(rng, shape=shape, dtype=jnp.bfloat16)


def make_weight(rng, shape):
    """Create a bf16 weight tensor with variance-preserving initialization.

    Uses std = 1/sqrt(fan_in) to preserve variance through the layer.
    """
    fan_in = shape[0]
    w = jax.random.normal(rng, shape=shape, dtype=jnp.bfloat16)
    return w / jnp.sqrt(jnp.array(fan_in, dtype=jnp.bfloat16))


def run_scenario(
    *,
    name,
    mesh,
    expected_fn,
    actual_fn,
    inputs,
    in_specs,
    out_spec,
    flops_per_device,
    benchmark_iter_counts=None,
):
    """Run a single scenario testing a Pallas kernel against a reference implementation.

    Args:
        name: Name of the scenario for display.
        mesh: JAX device mesh.
        expected_fn: Reference implementation function.
        actual_fn: Pallas kernel launch function (takes inputs + loop_iters kwarg).
        inputs: Tuple of input arrays.
        in_specs: Tuple of PartitionSpecs for inputs (None for replicated).
        out_spec: PartitionSpec for output (None for replicated).
        flops: Total FLOPs for one kernel execution (across all devices).
        benchmark_iter_counts: If provided, run benchmarking with these iteration counts.
    """
    print()
    print("-" * 60)
    print(name)
    print()

    # Convert None specs to replicated
    in_specs_resolved = tuple(
        PartitionSpec() if spec is None else spec for spec in in_specs
    )
    out_spec_resolved = PartitionSpec() if out_spec is None else out_spec

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=in_specs_resolved,
        out_specs=out_spec_resolved,
        check_vma=False,
    )
    def compute_expected(*args):
        return expected_fn(*args)

    expected = compute_expected(*inputs)

    if benchmark_iter_counts is not None:
        # Benchmarking mode: actual_fn takes (*inputs, loop_iters)
        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(*in_specs_resolved, PartitionSpec()),
            out_specs=out_spec_resolved,
            check_vma=False,
        )
        def compute_actual(*args_and_loop_iters):
            *args, loop_iters_arr = args_and_loop_iters
            return actual_fn(*args, loop_iters=loop_iters_arr)

        # Correctness check
        actual = compute_actual(*inputs, jnp.array(1, dtype=jnp.int32))
    else:
        # Simple mode (not used currently, but kept for flexibility)
        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=in_specs_resolved,
            out_specs=out_spec_resolved,
            check_vma=False,
        )
        def compute_actual(*args):
            return actual_fn(*args)

        actual = compute_actual(*inputs)

    assert (
        actual.shape == expected.shape
    ), f"actual shape: {actual.shape}, expected shape: {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"actual dtype: {actual.dtype}, expected dtype: {expected.dtype}"

    rel_rmse = float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))
    print(f"    rel_rmse: {rel_rmse:.3e}")

    if benchmark_iter_counts is not None:
        # don't spam logs with tens of thousands of debug prints
        if ENABLE_DEBUG:
            return

        if rel_rmse > 1e-2:
            print("    kernel output is incorrect; skipping benchmarking")
            return

        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(*in_specs_resolved, PartitionSpec()),
            out_specs=out_spec_resolved,
            check_vma=False,
        )
        def compute_calibration(*args_and_loop_iters):
            *args, loop_iters_arr = args_and_loop_iters
            return actual_fn(*args, loop_iters=loop_iters_arr, calibrate_only=True)

        # Warmup both actual and calibration
        warmup_iters = jnp.array(benchmark_iter_counts[0], dtype=jnp.int32)
        compute_actual(*inputs, warmup_iters).block_until_ready()
        compute_calibration(*inputs, warmup_iters).block_until_ready()

        # Collect timing data (kernel only = actual - calibration)
        kernel_times = []
        for n_iters in benchmark_iter_counts:
            n_iters_arr = jnp.array(n_iters, dtype=jnp.int32)

            start = time.perf_counter()
            compute_calibration(*inputs, n_iters_arr).block_until_ready()
            calibration_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            compute_actual(*inputs, n_iters_arr).block_until_ready()
            actual_elapsed = time.perf_counter() - start

            kernel_times.append(actual_elapsed - calibration_elapsed)

        # Linear regression
        iter_counts_arr = np.array(benchmark_iter_counts, dtype=np.float64)
        kernel_times_arr = np.array(kernel_times, dtype=np.float64)

        kernel_slope, _ = np.polyfit(iter_counts_arr, kernel_times_arr, deg=1)
        r_squared = np.corrcoef(iter_counts_arr, kernel_times_arr)[0, 1] ** 2

        print(f"    kernel run time: {kernel_slope * 1e6:.3f} us")
        if r_squared < 0.99:
            print(f"    WARNING: poor linear fit (R^2 = {r_squared:.4f})")

        # Compute effective TFLOP/s
        tflops = flops_per_device / (kernel_slope * 1e12)
        print(f"    effective bf16 TFLOP/s: {tflops:.3f}")


def create_2x2_mesh():
    devices = jax.devices()
    devices_by_coords = {tuple(d.coords): d for d in devices}
    assert devices_by_coords.keys() == {
        (i, j, 0) for i in range(2) for j in range(2)
    }, "expected a 2x2 device mesh"
    device_ring = [
        devices_by_coords[0, 0, 0],  # top-left
        devices_by_coords[0, 1, 0],  # top-right
        devices_by_coords[1, 1, 0],  # bottom-right
        devices_by_coords[1, 0, 0],  # bottom-left
    ]
    mesh = Mesh(device_ring, (AXIS_NAME,))
    return mesh


def main():
    mesh = create_2x2_mesh()

    BENCHMARK_ITER_COUNTS = [100, 500, 1000, 2000, 5000, 10000]
    N_LAYERS = 3

    rng = jax.random.key(0xCA7CAFE)

    # Sharding convention:
    # - Input activations: fully replicated
    # - Weights: sharded along the K2 dimension (K2 is a per-device quantity)
    #   - w1 [K1, K2*N_DEVICES] global -> [K1, K2] per device (sharded on dim 1)
    #   - w2 [K2*N_DEVICES, K1] global -> [K2, K1] per device (sharded on dim 0)

    # Scenario 1: Simple matmul
    # x: [N_BATCH, K1] replicated, w: [K1, K2*N_DEVICES] sharded -> out: [N_BATCH, K2*N_DEVICES] sharded
    rng, k1, k2 = jax.random.split(rng, 3)
    x_matmul = make_activation(k1, shape=(N_BATCH, K1))
    y_matmul = make_activation(k2, shape=(N_BATCH, K2 * N_DEVICES))
    w1_matmul = make_weight(k2, shape=(K1, K2 * N_DEVICES))
    w2_matmul = make_weight(k2, shape=(K2 * N_DEVICES, K1))

    run_scenario(
        name="matmul_w1",
        mesh=mesh,
        expected_fn=matmul_reference,
        actual_fn=matmul_pallas_launch,
        inputs=(x_matmul, w1_matmul),
        in_specs=(None, PartitionSpec(None, AXIS_NAME)),
        out_spec=PartitionSpec(None, AXIS_NAME),
        flops_per_device=2 * N_BATCH * K1 * K2,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
    )

    run_scenario(
        name="matmul_w2",
        mesh=mesh,
        expected_fn=matmul_reference,
        actual_fn=matmul_pallas_launch,
        inputs=(y_matmul, w2_matmul),
        in_specs=(PartitionSpec(None, AXIS_NAME), PartitionSpec(AXIS_NAME, None)),
        out_spec=PartitionSpec(None, AXIS_NAME),
        flops_per_device=2 * N_BATCH * K1 * K2,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
    )

    # Scenario 2: All-gather matmul
    # x: [N_BATCH, K1] sharded on K1 (dim 1) -> each device gets [N_BATCH, K1/N_DEVICES]
    # w1: [K1, K2*N_DEVICES] sharded on dim 1 -> each device gets [K1, K2]
    # out: [N_BATCH, K2*N_DEVICES] sharded on dim 1 -> each device gets [N_BATCH, K2]
    # The kernel gathers x along dim 1, then computes matmul
    rng, k1, k2 = jax.random.split(rng, 3)
    x_ag = make_activation(k1, shape=(N_BATCH, K1))
    w1_ag = make_weight(k2, shape=(K1, K2 * N_DEVICES))

    run_scenario(
        name="all_gather_matmul",
        mesh=mesh,
        expected_fn=all_gather_matmul_reference,
        actual_fn=all_gather_matmul_pallas_launch,
        inputs=(x_ag, w1_ag),
        in_specs=(PartitionSpec(None, AXIS_NAME), PartitionSpec(None, AXIS_NAME)),
        out_spec=PartitionSpec(None, AXIS_NAME),
        flops_per_device=2 * N_BATCH * K1 * K2,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
    )

    # Scenario 3: Matmul reduce-scatter
    # x: [N_BATCH, K2*N_DEVICES] sharded on dim 1 -> each device gets [N_BATCH, K2]
    # w2: [K2*N_DEVICES, K1] sharded on dim 0 -> each device gets [K2, K1]
    # out: [N_BATCH, K1] sharded on dim 1 -> each device gets [N_BATCH, K1/N_DEVICES]
    # The kernel computes local matmul (partial result), then reduce-scatters
    rng, k1, k2 = jax.random.split(rng, 3)
    x_rs = make_activation(k1, shape=(N_BATCH, K2 * N_DEVICES))
    w2_rs = make_weight(k2, shape=(K2 * N_DEVICES, K1))

    run_scenario(
        name="matmul_reduce_scatter",
        mesh=mesh,
        expected_fn=matmul_reduce_scatter_reference,
        actual_fn=matmul_reduce_scatter_pallas_launch,
        inputs=(x_rs, w2_rs),
        in_specs=(PartitionSpec(None, AXIS_NAME), PartitionSpec(AXIS_NAME, None)),
        out_spec=PartitionSpec(None, AXIS_NAME),
        flops_per_device=2 * N_BATCH * K2 * K1,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
    )

    # Scenario 4: Neural network (3 layers)
    # x: [N_BATCH, K1] replicated
    # w1s: [K1, K2*N_DEVICES] sharded on dim 1 -> each device gets [K1, K2]
    # w2s: [K2*N_DEVICES, K1] sharded on dim 0 -> each device gets [K2, K1]
    # Each layer: x @ w1 -> gelu -> x @ w2 -> all_reduce
    rng, k1 = jax.random.split(rng, 2)
    x_nn = make_activation(k1, shape=(N_BATCH, K1))

    w1_list = []
    w2_list = []
    for _ in range(N_LAYERS):
        rng, k1, k2 = jax.random.split(rng, 3)
        w1_list.append(make_weight(k1, shape=(K1, K2 * N_DEVICES)))
        w2_list.append(make_weight(k2, shape=(K2 * N_DEVICES, K1)))

    # FLOPs for neural network:
    # Per layer: matmul1 (2*N_BATCH*K1*K2*N_DEVICES) + gelu (~0) + matmul2 (2*N_BATCH*K2*N_DEVICES*K1)
    # Total: N_LAYERS * 2 * (2 * N_BATCH * K1 * K2 * N_DEVICES)
    nn_flops_per_device = N_LAYERS * 2 * (2 * N_BATCH * K1 * K2)

    run_scenario(
        name="neural_network",
        mesh=mesh,
        expected_fn=neural_network_reference,
        actual_fn=neural_network_pallas_launch,
        inputs=(x_nn, w1_list, w2_list),
        in_specs=(None, PartitionSpec(None, AXIS_NAME), PartitionSpec(AXIS_NAME, None)),
        out_spec=None,
        flops_per_device=nn_flops_per_device,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
    )

    print()


if __name__ == "__main__":
    main()
