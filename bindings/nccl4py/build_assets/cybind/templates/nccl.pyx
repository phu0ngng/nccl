# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

cimport cython  # NOQA
from libcpp.vector cimport vector

from ._internal.utils cimport (nested_resource, nullable_unique_ptr, get_buffer_pointer,
                              get_resource_ptr, get_nested_resource_ptr)

from enum import IntEnum as _IntEnum


$snippet_auto_lowpp_imports_pyx


###############################################################################
# POD
###############################################################################

cdef class PointerBox:
    """Stable storage for NCCL APIs that fill pointer outputs asynchronously."""

    def __init__(self, intptr_t ptr=0):
        self.ptr = ptr

    def __int__(self):
        return self.ptr

    def __index__(self):
        return self.ptr

    def __bool__(self):
        return self.ptr != 0

    @property
    def address(self):
        return <intptr_t>&self.ptr

    def __repr__(self):
        return f"<PointerBox ptr={self.ptr:#x}>"

    def __format__(self, format_spec):
        return format(self.ptr, format_spec)


$pod_defs


###############################################################################
# Enum
###############################################################################

$enum_defs


###############################################################################
# Error handling
###############################################################################

class NCCLError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value}): {get_error_string(status)}"
        super(NCCLError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != Result.Success and status != Result.InProgress:
        raise NCCLError(status)


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs
