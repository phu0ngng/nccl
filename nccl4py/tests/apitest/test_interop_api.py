"""API tests for interop module - requires GPU."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_resolve_tensor_basic():
    """Test resolve_tensor with PyTorch tensors on GPU."""
    import nccl.core.interop.torch as nccl_torch
    from nccl.core.typing import FLOAT32, INT64, BFLOAT16
    from cuda.core.experimental import Device

    # Set device
    device = Device(0)
    device.set_current()

    # Test with float32 tensor
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda:0')
    ptr, count, dtype, device_id = nccl_torch.resolve_tensor(tensor)

    assert isinstance(ptr, int)
    assert ptr > 0  # Valid pointer
    assert count == 3
    assert dtype is FLOAT32
    assert device_id == 0

    # Test with different shape and dtype
    tensor2 = torch.zeros((2, 5), dtype=torch.int64, device='cuda:0')
    ptr2, count2, dtype2, device_id2 = nccl_torch.resolve_tensor(tensor2)

    assert count2 == 10  # 2 * 5
    assert dtype2 is INT64

    # Test with bfloat16
    tensor3 = torch.zeros(100, dtype=torch.bfloat16, device='cuda:0')
    ptr3, count3, dtype3, device_id3 = nccl_torch.resolve_tensor(tensor3)

    assert count3 == 100
    assert dtype3 is BFLOAT16


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_resolve_tensor_aliases():
    """Test resolve_tensor with PyTorch dtype aliases."""
    import nccl.core.interop.torch as nccl_torch
    from nccl.core.typing import FLOAT16, FLOAT32, FLOAT64, INT64
    from cuda.core.experimental import Device

    device = Device(0)
    device.set_current()

    # Test with half (alias for float16)
    tensor = torch.zeros(10, dtype=torch.half, device='cuda:0')
    _, _, dtype, _ = nccl_torch.resolve_tensor(tensor)
    assert dtype is FLOAT16

    # Test with float (alias for float32)
    tensor = torch.zeros(10, dtype=torch.float, device='cuda:0')
    _, _, dtype, _ = nccl_torch.resolve_tensor(tensor)
    assert dtype is FLOAT32

    # Test with double (alias for float64)
    tensor = torch.zeros(10, dtype=torch.double, device='cuda:0')
    _, _, dtype, _ = nccl_torch.resolve_tensor(tensor)
    assert dtype is FLOAT64

    # Test with long (alias for int64)
    tensor = torch.zeros(10, dtype=torch.long, device='cuda:0')
    _, _, dtype, _ = nccl_torch.resolve_tensor(tensor)
    assert dtype is INT64


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_resolve_array_basic():
    """Test resolve_array with CuPy arrays on GPU."""
    import nccl.core.interop.cupy as nccl_cupy
    from nccl.core.typing import FLOAT64, UINT32, INT8

    # Test with float64 array
    array = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float64)
    ptr, count, dtype, device_id = nccl_cupy.resolve_array(array)

    assert isinstance(ptr, int)
    assert ptr > 0  # Valid pointer
    assert count == 4
    assert dtype is FLOAT64
    assert isinstance(device_id, int)
    assert device_id >= 0

    # Test with different shape and dtype
    array2 = cp.zeros((3, 7), dtype=cp.uint32)
    ptr2, count2, dtype2, device_id2 = nccl_cupy.resolve_array(array2)

    assert count2 == 21  # 3 * 7
    assert dtype2 is UINT32

    # Test with int8
    array3 = cp.ones((5, 4, 3), dtype=cp.int8)
    ptr3, count3, dtype3, device_id3 = nccl_cupy.resolve_array(array3)

    assert count3 == 60  # 5 * 4 * 3
    assert dtype3 is INT8


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_resolve_array_with_ml_dtypes():
    """Test resolve_array with ml-dtypes."""
    try:
        import ml_dtypes
    except ImportError:
        pytest.skip("ml-dtypes not installed")

    import nccl.core.interop.cupy as nccl_cupy
    from nccl.core.typing import BFLOAT16, FLOAT8E4M3, FLOAT8E5M2

    # Test with bfloat16
    array = cp.zeros(50, dtype=np.dtype("bfloat16"))
    ptr, count, dtype, device_id = nccl_cupy.resolve_array(array)

    assert count == 50
    assert dtype is BFLOAT16

    # Test with float8_e4m3fn
    array2 = cp.zeros(100, dtype=np.dtype("float8_e4m3fn"))
    ptr2, count2, dtype2, device_id2 = nccl_cupy.resolve_array(array2)

    assert count2 == 100
    assert dtype2 is FLOAT8E4M3

    # Test with float8_e5m2
    array3 = cp.zeros(100, dtype=np.dtype("float8_e5m2"))
    ptr3, count3, dtype3, device_id3 = nccl_cupy.resolve_array(array3)

    assert count3 == 100
    assert dtype3 is FLOAT8E5M2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_torch_empty_allocates_with_nccl():
    """Test that nccl.torch.empty() allocates memory with NCCL allocator."""
    import nccl.core.interop.torch as nccl_torch
    from cuda.core.experimental import Device

    device = Device(0)
    device.set_current()

    # Allocate tensor via NCCL
    tensor = nccl_torch.empty(100, dtype=torch.float32, device='cuda:0')

    assert tensor.shape == (100,)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == 'cuda'
    assert tensor.device.index == 0

    # Test multi-dimensional
    tensor2 = nccl_torch.empty(10, 20, dtype=torch.int32, device='cuda:0')
    assert tensor2.shape == (10, 20)
    assert tensor2.dtype == torch.int32


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_empty_allocates_with_nccl():
    """Test that nccl.cupy.empty() allocates memory with NCCL allocator."""
    import nccl.core.interop.cupy as nccl_cupy

    # Allocate array via NCCL
    array = nccl_cupy.empty(100, dtype='float32')

    assert array.shape == (100,)
    assert array.dtype == np.dtype('float32')

    # Test multi-dimensional
    array2 = nccl_cupy.empty((10, 20), dtype='int64')
    assert array2.shape == (10, 20)
    assert array2.dtype == np.dtype('int64')

    # Test with C and F order
    array_c = nccl_cupy.empty((5, 5), dtype='float64', order='C')
    assert array_c.flags['C_CONTIGUOUS']

    array_f = nccl_cupy.empty((5, 5), dtype='float64', order='F')
    assert array_f.flags['F_CONTIGUOUS']

