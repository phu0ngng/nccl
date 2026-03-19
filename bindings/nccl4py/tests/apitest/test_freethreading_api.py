"""
Free-threading stress tests for the nccl4py API layer.
"""

import pytest

import nccl.bindings as nccl_bindings


# Communicator property reads

@pytest.mark.mpi
@pytest.mark.parallel_threads_limit(16)
def test_comm_properties_concurrent(nccl_comm, rank_info):
    """Concurrent reads of communicator properties must be race-free."""
    assert nccl_comm.nranks == rank_info.nccl_size
    assert nccl_comm.rank == rank_info.nccl_rank
    assert nccl_comm.device.device_id == rank_info.nccl_local_rank


# NCCL memory allocation / deallocation

@pytest.mark.mpi
@pytest.mark.parallel_threads_limit(8)
def test_nccl_mem_alloc_free_concurrent(nccl_comm, rank_info):
    """
    Concurrent ncclMemAlloc / ncclMemFree calls exercise the nogil wrappers in
    cynccl.pyx directly. Multiple Python threads must be able to allocate and
    free device memory simultaneously without deadlocking or corrupting state.
    """
    ptr = nccl_bindings.mem_alloc(1024)
    assert ptr != 0
    nccl_bindings.mem_free(ptr)
