import numpy as np
import ml_dtypes
import pytest

import nccl.core as core
from nccl.core import typing as t
from nccl.core.typing import NcclInvalid


def test_nccldatatype_from_numpy_and_int_and_size():
    # From numpy dtype - standard types
    assert int(t.NcclDataType(np.dtype("float16"))) == int(t.FLOAT16.value)
    assert int(t.NcclDataType(np.dtype("float32"))) == int(t.FLOAT32.value)
    assert int(t.NcclDataType(np.dtype("float64"))) == int(t.FLOAT64.value)
    assert int(t.NcclDataType(np.dtype("int8"))) == int(t.INT8.value)
    assert int(t.NcclDataType(np.dtype("uint8"))) == int(t.UINT8.value)
    assert int(t.NcclDataType(np.dtype("int32"))) == int(t.INT32.value)
    assert int(t.NcclDataType(np.dtype("int64"))) == int(t.INT64.value)
    assert int(t.NcclDataType(np.dtype("uint32"))) == int(t.UINT32.value)
    assert int(t.NcclDataType(np.dtype("uint64"))) == int(t.UINT64.value)

    # From numpy dtype - ml-dtypes (optional)
    assert int(t.NcclDataType(np.dtype("bfloat16"))) == int(t.BFLOAT16.value)
    assert int(t.NcclDataType(np.dtype("float8_e4m3fn"))) == int(t.FLOAT8E4M3.value)
    assert int(t.NcclDataType(np.dtype("float8_e5m2"))) == int(t.FLOAT8E5M2.value)

    # From integer value (using existing constant value)
    dt_val = int(t.FLOAT16.value)
    assert int(t.NcclDataType(dt_val)) == dt_val

    # itemsize property - all types
    assert t.INT8.itemsize == 1
    assert t.UINT8.itemsize == 1
    assert t.FLOAT16.itemsize == 2
    assert t.BFLOAT16.itemsize == 2
    assert t.FLOAT32.itemsize == 4
    assert t.INT32.itemsize == 4
    assert t.UINT32.itemsize == 4
    assert t.FLOAT64.itemsize == 8
    assert t.INT64.itemsize == 8
    assert t.UINT64.itemsize == 8
    assert t.FLOAT8E4M3.itemsize == 1
    assert t.FLOAT8E5M2.itemsize == 1


def test_nccldatatype_to_numpy_dtype():
    """Test conversion from NcclDataType to numpy dtype."""
    # Standard types - always work
    assert t.INT8.numpy_dtype == np.dtype("int8")
    assert t.CHAR.numpy_dtype == np.dtype("int8")
    assert t.UINT8.numpy_dtype == np.dtype("uint8")
    assert t.INT32.numpy_dtype == np.dtype("int32")
    assert t.INT.numpy_dtype == np.dtype("int32")
    assert t.UINT32.numpy_dtype == np.dtype("uint32")
    assert t.INT64.numpy_dtype == np.dtype("int64")
    assert t.UINT64.numpy_dtype == np.dtype("uint64")
    assert t.FLOAT16.numpy_dtype == np.dtype("float16")
    assert t.HALF.numpy_dtype == np.dtype("float16")
    assert t.FLOAT32.numpy_dtype == np.dtype("float32")
    assert t.FLOAT.numpy_dtype == np.dtype("float32")
    assert t.FLOAT64.numpy_dtype == np.dtype("float64")
    assert t.DOUBLE.numpy_dtype == np.dtype("float64")

def test_nccldatatype_ml_dtypes_with_import():
    """Test ml-dtypes support."""

    # Verify numpy dtype creation works and check dtype.name
    assert np.dtype("bfloat16").name == "bfloat16"
    assert np.dtype("float8_e4m3fn").name == "float8_e4m3fn"  # Name matches construction string
    assert np.dtype("float8_e5m2").name == "float8_e5m2"

    # Verify NcclDataType.numpy_dtype works
    assert t.BFLOAT16.numpy_dtype == np.dtype("bfloat16")
    assert t.FLOAT8E4M3.numpy_dtype == np.dtype("float8_e4m3fn")
    assert t.FLOAT8E5M2.numpy_dtype == np.dtype("float8_e5m2")

    # Verify NcclDataType construction from ml-dtypes works and equals global constants
    assert int(t.NcclDataType(np.dtype("bfloat16"))) == int(t.BFLOAT16.value)
    assert int(t.NcclDataType(np.dtype("float8_e4m3fn"))) == int(t.FLOAT8E4M3.value)
    assert int(t.NcclDataType(np.dtype("float8_e5m2"))) == int(t.FLOAT8E5M2.value)

    # Verify equality (not identity, since constructor creates new instances)
    assert t.NcclDataType(np.dtype("bfloat16")) == t.BFLOAT16
    assert t.NcclDataType(np.dtype("float8_e4m3fn")) == t.FLOAT8E4M3
    assert t.NcclDataType(np.dtype("float8_e5m2")) == t.FLOAT8E5M2


def test_nccldatatype_equality_and_hash_and_repr():
    a = t.NcclDataType(np.dtype("float32"))
    b = t.NcclDataType(int(t.FLOAT32.value))
    assert a == b
    assert hash(a) == hash(b)
    assert "NcclDataType(" in repr(a)


def test_nccldatatype_invalid_dtype_raises():
    with pytest.raises(NcclInvalid):
        t.NcclDataType(np.dtype("complex64"))


def test_redop_valid_and_invalid():
    # Test valid RedOps
    for op_value, op_name in [
        (0, "sum"),
        (1, "prod"),
        (2, "max"),
        (3, "min"),
        (4, "avg"),
    ]:
        op = t.NcclRedOp(op_value)
        # int(NcclRedOp) should match int(core.OP)
        assert int(op) == op_value
        # .value property should match int(core.OP.value)
        assert op.value == op_value
        # .name property should match expected name (case-insensitive)
        assert op.name.lower() == op_name
        # __str__ and __repr__ should contain the name
        assert op_name in str(op).lower()
        assert op_name in repr(op).lower()
        # NcclRedOp should be equal to itself
        assert int(t.NcclRedOp(op_value)) == op_value
        # NcclRedOp should be usable in hash
        assert isinstance(hash(op), int)

    # Test that int(NcclRedOp) and int(NcclRedOp.value) are the same
    op = t.NcclRedOp(core.SUM.value)
    assert int(op) == int(op.value)

    # Test invalid RedOp raises NcclInvalid
    with pytest.raises(NcclInvalid):
        t.NcclRedOp(999999)

    # Test that NcclRedOp constructed from NcclRedOp.value is idempotent
    for op_value in [core.SUM.value, core.PROD.value, core.MAX.value, core.MIN.value, core.AVG.value]:
        op1 = t.NcclRedOp(op_value)
        op2 = t.NcclRedOp(op1.value)
        assert int(op1) == int(op2)
        assert op1.name == op2.name
        assert op1.value == op2.value


def test_public_constants_exposed_from_core():
    # Data types
    for name in [
        "INT8","CHAR","UINT8","INT32","INT","UINT32","INT64","UINT64",
        "FLOAT16","HALF","FLOAT32","FLOAT","FLOAT64","DOUBLE","BFLOAT16",
        "FLOAT8E4M3","FLOAT8E5M2",
    ]:
        assert hasattr(core, name)

    # RedOps
    for name in ["SUM","PROD","MAX","MIN","AVG"]:
        assert hasattr(core, name)
