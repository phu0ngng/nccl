# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated $version_span. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_${libname}_init = False

$wrapper_init


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("lib${libname}.so.2", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen lib${libname} ({err_msg.decode()})')
    return handle


cdef int _check_or_init_${libname}() except -1 nogil:
    global __py_${libname}_init
    if __py_${libname}_init:
        return 0

    # Load function
    cdef void* handle = NULL
${set_wrapper}

    __py_${libname}_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_${libname}()
    cdef dict data = {}

${set_functor}

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

$wrapper_defs
