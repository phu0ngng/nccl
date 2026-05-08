# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cy${libname} cimport *


###############################################################################
# Types
###############################################################################

$type_decls

ctypedef cudaStream_t Stream

cdef class PointerBox:
    cdef public intptr_t ptr


###############################################################################
# Enum
###############################################################################

$enum_decls


###############################################################################
# Functions
###############################################################################

$func_decls
