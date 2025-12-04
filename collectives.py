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
# Part 1: Implementing collectives in Pallas

## <--- your code here --->


def exchange_with_neighbor_pallas_scratch_specs(x):
    """
    Arguments:

    * `x`: Traced input array. You can use `x.shape` to query its shape at trace time.
    """

    # Returns a dictionary mapping string identifiers to Pallas scratch resource specifications.
    # These scratch resources will be allocated at the beginning of your kernel and passed in via
    # the `scratch_refs` argument.
    #
    # Types of scratch resources you can declare include:
    #
    # * Multi-dimensional arrays in VMEM: `pltpu.VMEM(shape=(...), dtype=...)`
    #   (example: `pltpu.VMEM(shape=(4, 8, 128), dtype=jnp.float32)`)
    #
    # * DMA semaphores: `pltpu.SemaphoreType.DMA(shape=(...))`, where `shape` is optional
    #   (example: `pltpu.SemaphoreType.DMA(shape=(2,))` for an array of 2 semaphores, or
    #   `pltpu.SemaphoreType.DMA` for a single semaphore)
    #
    # * Regular semaphores: `pltpu.SemaphoreType.REGULAR(shape=(...))`, where `shape` is optional
    #   (example: `pltpu.SemaphoreType.REGULAR(shape=(2,))` for an array of 2 semaphores, or
    #   `pltpu.SemaphoreType.REGULAR` for a single semaphore)
    #
    
    shape = x.shape
    # jax.debug.print("shape: {shape}", shape=shape)
    return {
        "dma_sems": pltpu.SemaphoreType.DMA(shape=(2,)),
    }


def exchange_with_neighbor_pallas_kernel(x_ref, out_ref, scratch_refs):
    """
    Exchanges data between neighboring devices according to the following pattern:
    * Device 0 sends its input array to device 1
    * Device 1 sends its input array to device 0
    * Device 2 sends its input array to device 3
    * Device 3 sends its input array to device 2

    Arguments:
    * `x_ref`: Pallas array ref pointing to the input array in VMEM. Read-only.
    * `out_ref`: Pallas array ref pointing to the output array in VMEM. Should be written to.
    * `scratch_refs`: Dictionary mapping string identifiers to preallocated scratch resources.
      The set of resources allocated is determined by your implementation of
      `exchange_with_neighbor_pallas_scratch_specs`.
    """

    # jax.debug.print("device {id}", id=pallas_get_my_device_id())

    neighbor_id = pallas_get_my_device_id() ^ 1
    # jax.debug.print("exchanging with device {neighbor_id}", neighbor_id=neighbor_id)

    dma_sems = scratch_refs["dma_sems"]
    src_sem = dma_sems.at[0]
    dst_sem = dma_sems.at[1]

    pallas_rdma_start(
        src_ref=x_ref,
        dst_ref=out_ref,
        dst_device_id=neighbor_id,
        src_send_sem=src_sem,
        dst_recv_sem=dst_sem,
    )

    pallas_rdma_wait_send(
        src_ref=x_ref, 
        src_send_sem=src_sem
    )
    pallas_rdma_wait_recv(
        dst_ref=out_ref, 
        dst_recv_sem=dst_sem
        )
    

def reduce_scatter_pallas_scratch_specs(x):
    """
    Arguments:

    * `x`: Traced input array. You can use `x.shape` to query its shape at trace time.
    """

    # Works the same way as the earlier scratch specs function
    # (see `exchange_with_neighbor_pallas_scratch_specs` above)

    num_rdmas = 32 # TODO: figure out actual num
    return {
        "vmem": pltpu.VMEM(shape=(x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype),
        "vmem2": pltpu.VMEM(shape=(x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype),

        "accumulation": pltpu.VMEM(shape=(x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype),
        "right_src_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
        "left_src_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
        
        "right_dst_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
        "left_dst_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
    }

def reduce_scatter_pallas_kernel_small(x_ref, out_ref, scratch_refs):

    id = pallas_get_my_device_id()
    array_size = x_ref.shape[0] // 4

    shards = 2
    shard_size = array_size // shards

    right_src_sems = scratch_refs["right_src_dma_sems"]
    right_dst_sems = scratch_refs["right_dst_dma_sems"]

    left_src_sems = scratch_refs["left_src_dma_sems"]
    left_dst_sems = scratch_refs["left_dst_dma_sems"]

    vmem1 = scratch_refs["vmem"]
    vmem2 = scratch_refs["vmem2"]
    accumulation = scratch_refs["accumulation"]
    right_neighbor = (id + 1) % 4
    left_neighbor = (id - 1) % 4

    # send missing data
    non_neighbor = (id + 2) % 4
    # send top shard right
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(non_neighbor * array_size, shard_size)],
        dst_ref=vmem1.at[pl.ds(non_neighbor * array_size, shard_size)], 
        dst_device_id=right_neighbor,
        src_send_sem=right_src_sems.at[0],
        dst_recv_sem=right_dst_sems.at[0]
    )
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(non_neighbor * array_size + shard_size, shard_size)],
        dst_ref=vmem1.at[pl.ds(non_neighbor * array_size + shard_size, shard_size)],
        dst_device_id=left_neighbor,
        src_send_sem=left_src_sems.at[0],
        dst_recv_sem=left_dst_sems.at[0]
    )

    # send bottom half of right neighbor 
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)],
        dst_ref=vmem1.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)], 
        dst_device_id=right_neighbor,
        src_send_sem=right_src_sems.at[1],
        dst_recv_sem=right_dst_sems.at[1]
    )
    # send top half of left neighbor
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(left_neighbor * array_size, shard_size)],
        dst_ref=vmem1.at[pl.ds(left_neighbor * array_size, shard_size)], 
        dst_device_id=left_neighbor,
        src_send_sem=left_src_sems.at[1],
        dst_recv_sem=left_dst_sems.at[1]
    )

    # initialize accumulation with own data
    accumulation[pl.ds(0, x_ref.shape[0])] = x_ref[pl.ds(0, x_ref.shape[0])]
    out_ref[pl.ds(0, array_size)] = x_ref[pl.ds(id*array_size, array_size)]

    # wait for data to forward to right neighbor
    pallas_rdma_wait_recv(
        dst_ref=vmem1.at[pl.ds(right_neighbor * array_size, shard_size)],
        dst_recv_sem=right_dst_sems.at[0]
    )
    accumulation[pl.ds(right_neighbor * array_size, shard_size)] += vmem1[pl.ds(right_neighbor * array_size, shard_size)]
    
    pallas_rdma_wait_recv(
        dst_ref=vmem1.at[pl.ds(left_neighbor * array_size + shard_size, shard_size)],
        dst_recv_sem=left_dst_sems.at[0]
    )
    accumulation[pl.ds(left_neighbor * array_size + shard_size, shard_size)] += vmem1[pl.ds(left_neighbor * array_size + shard_size, shard_size)]

    # phase 2: forward data to neighbors
    pallas_rdma_start(
        src_ref=accumulation.at[pl.ds(right_neighbor * array_size, shard_size)],
        dst_ref=vmem2.at[pl.ds(right_neighbor * array_size, shard_size)], # each core now has its own slice of vmem
        dst_device_id=right_neighbor,
        src_send_sem=right_src_sems.at[2],
        dst_recv_sem=right_dst_sems.at[2]
    )
    pallas_rdma_start(
        src_ref=accumulation.at[pl.ds(left_neighbor * array_size + shard_size, shard_size)],
        dst_ref=vmem2.at[pl.ds(left_neighbor * array_size + shard_size, shard_size)],
        dst_device_id=left_neighbor,
        src_send_sem=left_src_sems.at[2],
        dst_recv_sem=left_dst_sems.at[2]
    )

    # wait for values to arrive to id's array
    pallas_rdma_wait_recv(
        dst_ref=vmem1.at[pl.ds(id * array_size + shard_size, shard_size)], 
        dst_recv_sem=right_dst_sems.at[1]
    )
    pallas_rdma_wait_recv(
        dst_ref=vmem1.at[pl.ds(id * array_size, shard_size)], 
        dst_recv_sem=left_dst_sems.at[1]
    )
    out_ref[pl.ds(0, array_size)] += vmem1[pl.ds(id * array_size, array_size)] 

    pallas_rdma_wait_recv(
        dst_ref=vmem2.at[pl.ds(id * array_size, shard_size)], 
        dst_recv_sem=right_dst_sems.at[2]
    )    
    pallas_rdma_wait_recv(
        dst_ref=vmem2.at[pl.ds(id * array_size + shard_size, shard_size)], 
        dst_recv_sem=left_dst_sems.at[2]
    )

    out_ref[pl.ds(0, array_size)] += vmem2[pl.ds(id * array_size, array_size)] 

    # final wait sends
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(non_neighbor * array_size, shard_size)],
        src_send_sem=right_src_sems.at[0]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(non_neighbor * array_size + shard_size, shard_size)],
        src_send_sem=left_src_sems.at[0]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)],
        src_send_sem=right_src_sems.at[1]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(left_neighbor * array_size, shard_size)],
        src_send_sem=left_src_sems.at[1]
    )
    pallas_rdma_wait_send(
        src_ref=accumulation.at[pl.ds(right_neighbor * array_size, shard_size)],
        src_send_sem=right_src_sems.at[2]
    )
    pallas_rdma_wait_send(
        src_ref=accumulation.at[pl.ds(left_neighbor * array_size + shard_size, shard_size)],
        src_send_sem=left_src_sems.at[2]
    )


def reduce_scatter_pallas_kernel(x_ref, out_ref, scratch_refs):
    """
    Reduce-scatters data across the 4 devices in the ring, splitting the input array along its
    first dimension.

    Arguments:
    * `x_ref`: Input Pallas array ref in VMEM, of shape `(N, 8, 128)` where `N` is divisible by 16.
      Read-only.
    * `out_ref`: Output Pallas array ref in VMEM, of shape `(N // 4, 8, 128)`.
      Should be written to.
    * `scratch_refs`: Dictionary mapping string identifiers to preallocated scratch resources.
      The set of resources allocated is determined by your implementation of
      `reduce_scatter_pallas_scratch_specs`.
    """
    
    num_devices = 4
    # tips from lab: reorder operations
    id = pallas_get_my_device_id()
    array_size = x_ref.shape[0] // 4
    if (array_size > 128):
        shards = 16
    else:
        return reduce_scatter_pallas_kernel_small(
            x_ref, out_ref, scratch_refs
        )

    shard_size = array_size // shards
    shards_per_neighbor = shards // 2

    right_src_sems = scratch_refs["right_src_dma_sems"]
    right_dst_sems = scratch_refs["right_dst_dma_sems"]

    left_src_sems = scratch_refs["left_src_dma_sems"]
    left_dst_sems = scratch_refs["left_dst_dma_sems"]

    vmem = scratch_refs["vmem"]
    accumulation = scratch_refs["accumulation"]
    right_neighbor = (id + 1) % 4
    left_neighbor = (id - 1) % 4

    # prologue, start initial rdmas
    right_device_offset = (id - 1) % 4
    right_output_offset = right_device_offset * array_size

    left_device_offset = (id + 1) % 4
    left_output_offset = left_device_offset * array_size + shards_per_neighbor * shard_size

    for i in range(shards_per_neighbor):
        pallas_rdma_start(
            src_ref=x_ref.at[pl.ds(right_output_offset + i * shard_size, shard_size)],
            dst_ref=vmem.at[pl.ds(right_output_offset + i * shard_size, shard_size)],
            dst_device_id=right_neighbor,
            src_send_sem=right_src_sems.at[i],
            dst_recv_sem=right_dst_sems.at[i]
        )
        pallas_rdma_start(
            src_ref=x_ref.at[pl.ds(left_output_offset + i * shard_size, shard_size)],
            dst_ref=vmem.at[pl.ds(left_output_offset + i * shard_size, shard_size)],
            dst_device_id=left_neighbor,
            src_send_sem=left_src_sems.at[i],
            dst_recv_sem=left_dst_sems.at[i]
        )

    for i in range(1, num_devices-1):
        # cycle through top half to send to right neighbor
        right_device_offset = (id - i - 1) % 4
        right_output_offset = right_device_offset * array_size 
        # bottom half to send to left
        left_device_offset = (id + i + 1) % 4
        left_output_offset = left_device_offset * array_size + shard_size*shards_per_neighbor
          
        # wait for previous rdma to be written
        for shard_i in range(0, shards_per_neighbor, 1):
            pallas_rdma_wait_recv(
                dst_ref=vmem.at[pl.ds(right_output_offset + shard_i*shard_size, shard_size)],
                dst_recv_sem=right_dst_sems.at[(i-1)*shards_per_neighbor + shard_i]
            )
        
            accumulation[pl.ds(right_output_offset + shard_i*shard_size, shard_size)] = (
                x_ref[pl.ds(right_output_offset + shard_i*shard_size, shard_size)] 
                + vmem[pl.ds(right_output_offset + shard_i*shard_size, shard_size)]
            )
            pallas_rdma_start(
                src_ref=accumulation.at[pl.ds(right_output_offset + shard_i*shard_size, shard_size)],
                dst_ref=vmem.at[pl.ds(right_output_offset + shard_i*shard_size, shard_size)],
                dst_device_id=right_neighbor,
                src_send_sem=right_src_sems.at[i*shards_per_neighbor+shard_i],
                dst_recv_sem=right_dst_sems.at[i*shards_per_neighbor+shard_i]
            )
            pallas_rdma_wait_recv(
                dst_ref=vmem.at[pl.ds(left_output_offset + shard_i*shard_size, shard_size)],
                dst_recv_sem=left_dst_sems.at[(i-1)*shards_per_neighbor + shard_i]
            )
            accumulation[pl.ds(left_output_offset + shard_i*shard_size, shard_size)] = (
                x_ref[pl.ds(left_output_offset + shard_i*shard_size, shard_size)] 
                + vmem[pl.ds(left_output_offset + shard_i*shard_size, shard_size)]
            )
            pallas_rdma_start(
                src_ref=accumulation.at[pl.ds(left_output_offset + shard_i*shard_size, shard_size)],
                dst_ref=vmem.at[pl.ds(left_output_offset + shard_i*shard_size, shard_size)],
                dst_device_id=left_neighbor,
                src_send_sem=left_src_sems.at[i*shards_per_neighbor + shard_i],
                dst_recv_sem=left_dst_sems.at[i*shards_per_neighbor + shard_i]
            )

    # all wait_sends at the end, for some reason this faster
    for i in range(0, num_devices-1):
        right_device_offset = (id - i - 1) % 4
        right_output_offset = right_device_offset * array_size 
        # bottom half to send to left
        left_device_offset = (id + i + 1) % 4
        left_output_offset = left_device_offset * array_size + shard_size*shards_per_neighbor
        for shard_i in range(shards_per_neighbor):
            pallas_rdma_wait_send(
                src_ref=accumulation.at[pl.ds(right_output_offset + shard_i*shard_size, shard_size)],
                src_send_sem=right_src_sems.at[i*shards_per_neighbor + shard_i]
            )
            pallas_rdma_wait_send(
                src_ref=accumulation.at[pl.ds(left_output_offset+ shard_i*shard_size, shard_size)],
                src_send_sem=left_src_sems.at[i*shards_per_neighbor + shard_i]
            )
    
    # epilogue, wait for the last rdmas to arrive
    final_offset = (id) * array_size
    for shard_i in range(shards_per_neighbor):
        pallas_rdma_wait_recv(
            dst_ref=vmem.at[pl.ds(final_offset + shard_i*shard_size, shard_size)],
            dst_recv_sem=right_dst_sems.at[(num_devices-2)*shards_per_neighbor +shard_i]
        )
        out_ref[pl.ds(shard_i*shard_size, shard_size)] = (
            vmem[pl.ds(final_offset + shard_i*shard_size, shard_size)] 
            + x_ref[pl.ds(final_offset+ shard_i*shard_size, shard_size)]
        )
        pallas_rdma_wait_recv(
            dst_ref=vmem.at[pl.ds(final_offset + shard_size * shards_per_neighbor + shard_i*shard_size, shard_size)],
            dst_recv_sem=left_dst_sems.at[(num_devices-2)*shards_per_neighbor + shard_i]
        )
        out_ref[pl.ds(shard_size * shards_per_neighbor+ shard_i * shard_size, shard_size)] = (
            vmem[pl.ds(final_offset + shard_size * shards_per_neighbor + shard_i * shard_size, shard_size)] 
            + x_ref[pl.ds(final_offset + shard_size * shards_per_neighbor + shard_i * shard_size, shard_size)])
    

def all_gather_pallas_scratch_specs(x):
    """
    Arguments:

    * `x`: Traced input array. You can use `x.shape` to query its shape at trace time.
    """

    # Works the same way as the earlier scratch specs function
    # (see `exchange_with_neighbor_pallas_scratch_specs` above)
    num_rdmas = 6 # TODO: figure out actual num

    return {
        # "vmem": pltpu.VMEM(shape=(x.shape[0] // 4, x.shape[1], x.shape[2]), dtype=x.dtype),
        "src_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
        "dst_dma_sems": pltpu.SemaphoreType.DMA(shape=(num_rdmas,)),
    }


def all_gather_pallas_kernel(x_ref, out_ref, scratch_refs):
    """
    All-gathers data across the 4 devices in the ring, concatenating along the first dimension.

    Arguments:
    * `x_ref`: Input Pallas array ref in VMEM, of shape `(N, 8, 128)`, where `N` is divisible by 4.
      Read-only.
    * `out_ref`: Output Pallas array ref in VMEM, of shape `(N * 4, 8, 128)`.
      Should be written to.
    * `scratch_refs`: Dictionary mapping string identifiers to preallocated scratch resources.
      The set of resources allocated is determined by your implementation of
      `all_gather_pallas_scratch_specs`.
    """

    # tips from lab: reorder operations
    id = pallas_get_my_device_id()
    array_size = x_ref.shape[0]
    shard_size = array_size // 2

    src_sems = scratch_refs["src_dma_sems"]
    dst_sems = scratch_refs["dst_dma_sems"]

    # send to right neighbor
    right_neighbor = (id + 1) % 4
    left_neighbor = (id - 1) % 4

    #TODO: make dma indices more clear?
    # split array in two, send as two rdmas to right neighbor
    # note: sending bottom half first since we need it to send to the left neighbor
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(shard_size, shard_size)],
        dst_ref=out_ref.at[pl.ds(id * array_size + shard_size, shard_size)], 
        dst_device_id=right_neighbor,
        src_send_sem=src_sems.at[0],
        dst_recv_sem=dst_sems.at[0]
    )
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(0, shard_size)],
        dst_ref=out_ref.at[pl.ds(id * array_size, shard_size)],
        dst_device_id=left_neighbor,
        src_send_sem=src_sems.at[2],
        dst_recv_sem=dst_sems.at[2]
    )
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(0, shard_size)],
        dst_ref=out_ref.at[pl.ds(id * array_size, shard_size)],
        dst_device_id=right_neighbor,
        src_send_sem=src_sems.at[1],
        dst_recv_sem=dst_sems.at[1]
    )
    pallas_rdma_start(
        src_ref=x_ref.at[pl.ds(shard_size, shard_size)],
        dst_ref=out_ref.at[pl.ds(id * array_size + shard_size, shard_size)],
        dst_device_id=left_neighbor,
        src_send_sem=src_sems.at[3],
        dst_recv_sem=dst_sems.at[3]
    )

    out_ref[pl.ds(id*array_size, array_size)] = x_ref[pl.ds(0, array_size)]

    # wait for the halves that need to be written to neighbors to be recieved
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)], 
        dst_recv_sem=dst_sems.at[0] # bottom half of rdma from right neighbor
    )
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(left_neighbor * array_size, shard_size)], 
        dst_recv_sem=dst_sems.at[2] # top half of rdma from left neighbor
    )    

    # # write missing tiles
    pallas_rdma_start(
        src_ref=out_ref.at[pl.ds(left_neighbor * array_size, shard_size)],
        dst_ref=out_ref.at[pl.ds(left_neighbor * array_size, shard_size)],
        dst_device_id=right_neighbor,
        src_send_sem=src_sems.at[4],
        dst_recv_sem=dst_sems.at[4]
    )
    pallas_rdma_start(
        src_ref=out_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)],
        dst_ref=out_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)],
        dst_device_id=left_neighbor,
        src_send_sem=src_sems.at[5],
        dst_recv_sem=dst_sems.at[5]
    )

    # wait for all rdma reads
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(shard_size, shard_size)], 
        src_send_sem=src_sems.at[0]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(0, shard_size)],
        src_send_sem=src_sems.at[1]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(0, shard_size)], 
        src_send_sem=src_sems.at[2]
    )
    pallas_rdma_wait_send(
        src_ref=x_ref.at[pl.ds(shard_size, shard_size)], 
        src_send_sem=src_sems.at[3]
    )
    pallas_rdma_wait_send(
        src_ref=out_ref.at[pl.ds(left_neighbor * array_size, shard_size)], 
        src_send_sem=src_sems.at[4]
    )
    pallas_rdma_wait_send(
        src_ref=out_ref.at[pl.ds(right_neighbor * array_size + shard_size, shard_size)], 
        src_send_sem=src_sems.at[5]
    )

    # wait for second phase of rdmas to finish writing
    missing = (id + 2) % 4
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(right_neighbor * array_size, shard_size)], 
        dst_recv_sem=dst_sems.at[1] # bottom half of rdma from right neighbor
    )
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(left_neighbor * array_size + shard_size, shard_size)], 
        dst_recv_sem=dst_sems.at[3] # top half of rdma from left neighbor
    )   
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(missing * array_size, shard_size)], 
        dst_recv_sem=dst_sems.at[4]
    )
    pallas_rdma_wait_recv(
        dst_ref=out_ref.at[pl.ds(missing * array_size+ shard_size, shard_size)], 
        dst_recv_sem=dst_sems.at[5]
    )


## <--- /your code here --->


################################################################################
##                YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.              ##
################################################################################


def exchange_with_neighbor_reference(x):
    return lax.ppermute(
        x,
        axis_name=AXIS_NAME,
        perm=[(0, 1), (1, 0), (2, 3), (3, 2)],
    )


def reduce_scatter_reference(x):
    return lax.psum_scatter(x, axis_name=AXIS_NAME, scatter_dimension=0, tiled=True)


def all_gather_reference(x):
    return lax.all_gather(x, axis_name=AXIS_NAME, axis=0, tiled=True)


def exchange_with_neighbor_pallas_launch(x):
    assert x.ndim == 3
    assert x.shape[-2:] == (8, 128)
    assert x.dtype == jnp.float32

    return pl.pallas_call(
        kernel=exchange_with_neighbor_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        scratch_shapes=(exchange_with_neighbor_pallas_scratch_specs(x),),
        in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        interpret=get_interpret_params(),
    )(x)


def reduce_scatter_pallas_launch(x, loop_iters, calibrate_only=False):
    assert x.ndim == 3
    assert x.shape[-2:] == (8, 128)
    assert x.dtype == jnp.float32

    out_shape = (x.shape[0] // N_DEVICES, x.shape[1], x.shape[2])

    return launch_kernel_looped(
        kernel=reduce_scatter_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=out_shape, dtype=x.dtype),
        scratch_shapes=reduce_scatter_pallas_scratch_specs(x),
        loop_iters=loop_iters,
        input_array=x,
        calibrate_only=calibrate_only,
    )


def all_gather_pallas_launch(x, loop_iters, calibrate_only=False):
    assert x.ndim == 3
    assert x.shape[-2:] == (8, 128)
    assert x.dtype == jnp.float32

    out_shape = (x.shape[0] * N_DEVICES, x.shape[1], x.shape[2])

    return launch_kernel_looped(
        kernel=all_gather_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=out_shape, dtype=x.dtype),
        scratch_shapes=all_gather_pallas_scratch_specs(x),
        loop_iters=loop_iters,
        input_array=x,
        calibrate_only=calibrate_only,
    )

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
    input_array,
    calibrate_only=False,
):
    def kernel_wrapper(x_ref, loop_iters_ref, out_ref, loop_sems, scratch_refs):
        my_id = pallas_get_my_device_id()
        other_devices = [
            lax.rem(my_id + offset, N_DEVICES) for offset in range(1, N_DEVICES)
        ]

        def loop_body(_, __):
            out_ref[...] = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)
            pl.delay(1)  # optimization barrier

            if not calibrate_only:
                kernel(x_ref, out_ref, scratch_refs)

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
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20),
        interpret=get_interpret_params(),
    )(input_array, loop_iters_arr)


def run_scenario(
    *,
    name,
    mesh,
    expected_fn,
    actual_fn,
    input_shapes,
    input_dtype,
    benchmark_iter_counts=None,
    bytes_transferred_fn=None,
):
    print()
    print("-" * 60)
    print(name)

    for input_shape in input_shapes:
        print()
        print(f"    running with per-device input shape: {input_shape}")

        @jax.jit
        @jax.shard_map(
            mesh=mesh,
            in_specs=(PartitionSpec(AXIS_NAME),),
            out_specs=PartitionSpec(AXIS_NAME),
            check_vma=False,
        )
        def compute_expected(x):
            expected = expected_fn(x)
            return lax.all_gather(expected, axis_name=AXIS_NAME, tiled=True)

        rng = jax.random.key(0xCA7CAFE)
        x = jax.random.normal(
            rng,
            shape=(input_shape[0] * N_DEVICES, *input_shape[1:]),
            dtype=input_dtype,
        )
        expected = compute_expected(x)

        if benchmark_iter_counts is not None:
            # Benchmarking mode: actual_fn takes (x, loop_iters)
            @jax.jit
            @jax.shard_map(
                mesh=mesh,
                in_specs=(PartitionSpec(AXIS_NAME), PartitionSpec()),
                out_specs=PartitionSpec(AXIS_NAME),
                check_vma=False,
            )
            def compute_actual(x, loop_iters_arr):
                actual = actual_fn(x, loop_iters=loop_iters_arr)
                return lax.all_gather(actual, axis_name=AXIS_NAME, tiled=True)

            # Correctness check
            actual = compute_actual(x, jnp.array(1, dtype=jnp.int32))
        else:
            # Simple mode: actual_fn takes just x
            @jax.jit
            @jax.shard_map(
                mesh=mesh,
                in_specs=(PartitionSpec(AXIS_NAME),),
                out_specs=PartitionSpec(AXIS_NAME),
                check_vma=False,
            )
            def compute_actual(x):
                actual = actual_fn(x)
                return lax.all_gather(actual, axis_name=AXIS_NAME, tiled=True)

            actual = compute_actual(x)

        assert (
            actual.shape == expected.shape
        ), f"actual shape: {actual.shape}, expected shape: {expected.shape}"
        assert (
            actual.dtype == expected.dtype
        ), f"actual dtype: {actual.dtype}, expected dtype: {expected.dtype}"

        rel_rmse = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
        print(f"    rel_rmse: {rel_rmse:.3e}")

        if benchmark_iter_counts is not None:
            # don't spam logs with tens of thousands of debug prints
            if ENABLE_DEBUG:
                continue

            if rel_rmse > 1e-7:
                print("    kernel output is incorrect; skipping benchmarking")
                continue

            @jax.jit
            @jax.shard_map(
                mesh=mesh,
                in_specs=(PartitionSpec(AXIS_NAME), PartitionSpec()),
                out_specs=PartitionSpec(AXIS_NAME),
                check_vma=False,
            )
            def compute_calibration(x, loop_iters_arr):
                actual = actual_fn(x, loop_iters=loop_iters_arr, calibrate_only=True)
                return lax.all_gather(actual, axis_name=AXIS_NAME, tiled=True)

            # Warmup both actual and calibration
            warmup_iters = jnp.array(benchmark_iter_counts[0], dtype=jnp.int32)
            compute_actual(x, warmup_iters).block_until_ready()
            compute_calibration(x, warmup_iters).block_until_ready()

            # Collect timing data (kernel only = actual - calibration)
            kernel_times = []
            for n_iters in benchmark_iter_counts:
                n_iters_arr = jnp.array(n_iters, dtype=jnp.int32)

                start = time.perf_counter()
                compute_calibration(x, n_iters_arr).block_until_ready()
                calibration_elapsed = time.perf_counter() - start

                start = time.perf_counter()
                compute_actual(x, n_iters_arr).block_until_ready()
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

            if bytes_transferred_fn is not None:
                bytes_transferred = bytes_transferred_fn(
                    input_shape=input_shape,
                    input_dtype=input_dtype,
                )
                bandwidth = bytes_transferred / kernel_slope
                print(f"    effective bandwidth: {bandwidth / 1e9:.3f} GB/s")


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


def bytes_transferred_reduce_scatter(
    *,
    input_shape,
    input_dtype,
):
    total_elems = 1
    for dim in input_shape:
        total_elems *= dim
    bytes_per_elem = jnp.dtype(input_dtype).itemsize
    return bytes_per_elem * total_elems * (N_DEVICES - 1) / N_DEVICES


def bytes_transferred_all_gather(
    *,
    input_shape,
    input_dtype,
):
    # In all-gather, each device receives (N_DEVICES - 1) shards from the network
    total_elems = 1
    for dim in input_shape:
        total_elems *= dim
    bytes_per_elem = jnp.dtype(input_dtype).itemsize
    return bytes_per_elem * total_elems * (N_DEVICES - 1)


def main():
    mesh = create_2x2_mesh()

    BENCHMARK_ITER_COUNTS = [100, 500, 1000, 2000, 5000, 10000]

    run_scenario(
        name="exchange_with_neighbor",
        mesh=mesh,
        expected_fn=exchange_with_neighbor_reference,
        actual_fn=exchange_with_neighbor_pallas_launch,
        input_shapes=((1024, 8, 128),),
        input_dtype=jnp.float32,
    )

    run_scenario(
        name="reduce_scatter",
        mesh=mesh,
        expected_fn=reduce_scatter_reference,
        actual_fn=reduce_scatter_pallas_launch,
        input_shapes=(
            (16, 8, 128),
            (128, 8, 128),
            (1024, 8, 128),
            (2048, 8, 128),
            (4096, 8, 128),
        ),
        input_dtype=jnp.float32,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
        bytes_transferred_fn=bytes_transferred_reduce_scatter,
    )

    run_scenario(
        name="all_gather",
        mesh=mesh,
        expected_fn=all_gather_reference,
        actual_fn=all_gather_pallas_launch,
        input_shapes=(
            (4, 8, 128),
            (32, 8, 128),
            (256, 8, 128),
            (512, 8, 128),
            (1024, 8, 128),
        ),
        input_dtype=jnp.float32,
        benchmark_iter_counts=BENCHMARK_ITER_COUNTS,
        bytes_transferred_fn=bytes_transferred_all_gather,
    )

    print()


if __name__ == "__main__":
    main()
