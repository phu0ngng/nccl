"""Unit tests for interop module dtype conversions."""
import pytest
import ml_dtypes
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
def test_torch_to_nccl_dtype():
    """Test torch.to_nccl_dtype conversion returns global constants."""
    import nccl.core.interop.torch as nccl_torch
    from nccl.core.typing import (
        FLOAT16, FLOAT32, FLOAT64, INT8, INT32, INT64, UINT8, BFLOAT16,
        FLOAT8E4M3, FLOAT8E5M2
    )

    # Test that private function returns the exact global constant objects (identity check)
    assert nccl_torch._to_nccl_dtype(torch.float16) is FLOAT16
    assert nccl_torch._to_nccl_dtype(torch.float32) is FLOAT32
    assert nccl_torch._to_nccl_dtype(torch.float64) is FLOAT64
    assert nccl_torch._to_nccl_dtype(torch.int8) is INT8
    assert nccl_torch._to_nccl_dtype(torch.int32) is INT32
    assert nccl_torch._to_nccl_dtype(torch.int64) is INT64
    assert nccl_torch._to_nccl_dtype(torch.uint8) is UINT8
    assert nccl_torch._to_nccl_dtype(torch.bfloat16) is BFLOAT16

    # Test PyTorch aliases also return same global constants
    assert nccl_torch._to_nccl_dtype(torch.half) is FLOAT16
    assert nccl_torch._to_nccl_dtype(torch.float) is FLOAT32
    assert nccl_torch._to_nccl_dtype(torch.double) is FLOAT64
    assert nccl_torch._to_nccl_dtype(torch.long) is INT64
    assert nccl_torch._to_nccl_dtype(torch.bool) is UINT8

    assert nccl_torch._to_nccl_dtype(torch.float8_e4m3fn) is FLOAT8E4M3
    assert nccl_torch._to_nccl_dtype(torch.float8_e5m2) is FLOAT8E5M2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_torch_to_nccl_dtype_unsupported():
    """Test torch.to_nccl_dtype with unsupported types."""
    import nccl.core.interop.torch as nccl_torch
    from nccl.core.typing import NcclInvalid

    with pytest.raises(NcclInvalid, match="complex.*has no NCCL equivalent"):
        nccl_torch._to_nccl_dtype(torch.complex64)

    with pytest.raises(NcclInvalid, match="complex.*has no NCCL equivalent"):
        nccl_torch._to_nccl_dtype(torch.complex128)

    with pytest.raises(NcclInvalid, match="has no NCCL equivalent"):
        nccl_torch._to_nccl_dtype(torch.cfloat)

    with pytest.raises(NcclInvalid, match="has no NCCL equivalent"):
        nccl_torch._to_nccl_dtype(torch.int16)

    with pytest.raises(NcclInvalid, match="qint8.*has no NCCL equivalent"):
        nccl_torch._to_nccl_dtype(torch.qint8)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_to_nccl_dtype():
    """Test cupy.to_nccl_dtype conversion returns global constants."""
    import nccl.core.interop.cupy as nccl_cupy
    from nccl.core.typing import (
        FLOAT16, FLOAT32, FLOAT64, INT8, INT32, INT64, UINT8, UINT32, UINT64,
        BFLOAT16, FLOAT8E4M3, FLOAT8E5M2
    )

    # Test that private function returns the exact global constant objects (identity check)
    assert nccl_cupy._to_nccl_dtype(cp.float16) is FLOAT16
    assert nccl_cupy._to_nccl_dtype(cp.float32) is FLOAT32
    assert nccl_cupy._to_nccl_dtype(cp.float64) is FLOAT64
    assert nccl_cupy._to_nccl_dtype(cp.int8) is INT8
    assert nccl_cupy._to_nccl_dtype(cp.int32) is INT32
    assert nccl_cupy._to_nccl_dtype(cp.int64) is INT64
    assert nccl_cupy._to_nccl_dtype(cp.uint8) is UINT8
    assert nccl_cupy._to_nccl_dtype(cp.uint32) is UINT32
    assert nccl_cupy._to_nccl_dtype(cp.uint64) is UINT64

    # Also test with numpy dtype directly
    assert nccl_cupy._to_nccl_dtype(np.dtype("float32")) is FLOAT32

    assert nccl_cupy._to_nccl_dtype(np.dtype("bfloat16")) is BFLOAT16
    assert nccl_cupy._to_nccl_dtype(np.dtype("float8_e4m3fn")) is FLOAT8E4M3
    assert nccl_cupy._to_nccl_dtype(np.dtype("float8_e5m2")) is FLOAT8E5M2


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_to_nccl_dtype_unsupported():
    """Test cupy.to_nccl_dtype with unsupported numpy/cupy types."""
    import nccl.core.interop.cupy as nccl_cupy
    from nccl.core.typing import NcclInvalid

    # Test complex types - not supported by NCCL
    with pytest.raises(NcclInvalid, match="complex.*has no NCCL equivalent"):
        nccl_cupy._to_nccl_dtype(np.dtype("complex64"))

    with pytest.raises(NcclInvalid, match="complex.*has no NCCL equivalent"):
        nccl_cupy._to_nccl_dtype(np.dtype("complex128"))

    # Test int16/uint16 - not supported by NCCL
    with pytest.raises(NcclInvalid, match="does not support.*16-bit"):
        nccl_cupy._to_nccl_dtype(np.dtype("int16"))

    with pytest.raises(NcclInvalid, match="does not support.*16-bit"):
        nccl_cupy._to_nccl_dtype(np.dtype("uint16"))

    # Test object dtype
    with pytest.raises(NcclInvalid, match="object.*has no NCCL equivalent"):
        nccl_cupy._to_nccl_dtype(np.dtype("object"))

    # Test string dtype
    with pytest.raises(NcclInvalid, match="string.*has no NCCL equivalent"):
        nccl_cupy._to_nccl_dtype(np.dtype("U10"))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_torch_to_nccl_dtype_without_torch(monkeypatch):
    """Test that _to_nccl_dtype raises ModuleNotFoundError when torch not available."""
    # Temporarily disable torch
    import nccl.core.interop.torch as nccl_torch
    monkeypatch.setattr(nccl_torch, "_torch_enabled", False)

    with pytest.raises(ModuleNotFoundError, match="PyTorch is not installed"):
        nccl_torch._to_nccl_dtype("dummy")


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_to_nccl_dtype_without_cupy(monkeypatch):
    """Test that _to_nccl_dtype raises ModuleNotFoundError when cupy not available."""
    # Temporarily disable cupy
    import nccl.core.interop.cupy as nccl_cupy
    monkeypatch.setattr(nccl_cupy, "_cupy_enabled", False)

    with pytest.raises(ModuleNotFoundError, match="CuPy is not installed"):
        nccl_cupy._to_nccl_dtype("dummy")
