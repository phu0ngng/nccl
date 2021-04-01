/*
 * Copyright 2009-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/******************************************************************************
 *
 *   Module: cudamemcheck.h
 *
 *   Description:
 *       Shared definitions between memcheck front-end and back-end
 *
 ******************************************************************************/

#ifndef _CUDAMEMCHECK_H
#define _CUDAMEMCHECK_H

/* We share the CUDBGException_t type */
#include <cudadebugger.h>

/* Environment variables that are set by the toolkit

    CUDA_MEMCHECK_ENV_ENABLE => "CUDA_MEMCHECK"
    Set if memcheck should be enabled
        For toolkits older than 4.1, this is set to 1
        For CUDA 4.1 onwards, the value contained is the build number

    CUDA_MEMCHECK_ENV_OUTPUT => "CUDA_MEMCHECK_OUTPUT"
    Sets the output file name for the memcheck records
    v1 (TK_VERSION < 18) Used by 5.0 and older clients
        This contains the filename
    v2 (TK_VERSION >= 18) Used for 5.5 and above
        This variable is the following format :
        <CCIPCtype>:<PROP_NAME>=<PROP_VAL>{,<NAME2>=<VAL2>,...}
    CCIPCtype = {LEGACY, FILE, SHMEM}
        LEGACY:
            Setup properties via environment variables, output to file
            Properties :
                PROP_OUT_FILE = <file-name>
        FILE:
            Bootstrap and output via files
            Properties :
                PROP_OUT_FILE = <file_name>
                PROP_IN_FILE  = <file_name>
        SHMEM:
            Bootstrap and output via shmem
            Properties :
                PROP_TOOLKIT_PID = pid

    CUDA_MEMCHECK_ENV_FLAGS  => "CUDA_MEMCHECK_FLAGS"
    Bitvector of flags (see CUmemcheck_flags)

    CUDA_MEMCHECK_ENV_RECVER => "CUDA_MEMCHECK_RECORD_FORMAT_VERSION"
    Defines the record format version.
        Undefined for toolkits older than 4.0
        For CUDA 4.0 toolkit, set to 4
        For CUDA 4.1 toolkit, set to 7
        For CUDA 5.0 toolkit, set to 9

    CUDA_MEMCHECK_ENV_DYNSCHEME => "CUDA_MEMCHECK_DYNAMIC_SCHEME"
    String containing name of dynamic scheme to use
*/
#define CUDA_MEMCHECK_ENV_ENABLE    "CUDA_MEMCHECK"
#define CUDA_MEMCHECK_ENV_OUTPUT    "CUDA_MEMCHECK_OUTPUT"
#define CUDA_MEMCHECK_ENV_FLAGS     "CUDA_MEMCHECK_FLAGS"
#define CUDA_MEMCHECK_ENV_RECVER    "CUDA_MEMCHECK_RECORD_FORMAT_VERSION"
#define CUDA_MEMCHECK_ENV_DYNSCHEME "CUDA_MEMCHECK_DYNAMIC_SCHEME"

/* Version of current error record format. This is now a number */
#define CUDA_MEMCHECK_RECORD_VERSION    9

/* Version handling for cuda memcheck toolkit app.
   The major and minor numbers correspond to toolkit releases.
   The build number is a permanently incrementing number
*/
#define CUDA_MEMCHECK_TK_VERSION_MAJOR     10
#define CUDA_MEMCHECK_TK_VERSION_MINOR     0
#define CUDA_MEMCHECK_TK_VERSION_BUILD     46

/* Toolkit version cutoffs
   For backward compatibility, build == 1 is CUDA 4.0
   For CUDA 4.1 the numbering started at 2
*/
#define CUDA_MEMCHECK_TK_VERSION_4_0       1
#define CUDA_MEMCHECK_TK_VERSION_4_1       2
#define CUDA_MEMCHECK_TK_VERSION_4_2       3
#define CUDA_MEMCHECK_TK_VERSION_5_0       17
#define CUDA_MEMCHECK_TK_VERSION_5_5       25
#define CUDA_MEMCHECK_TK_VERSION_6_0       33
#define CUDA_MEMCHECK_TK_VERSION_6_5       36
#define CUDA_MEMCHECK_TK_VERSION_7_0       40
#define CUDA_MEMCHECK_TK_VERSION_8_0       41
#define CUDA_MEMCHECK_TK_VERSION_9_0       43
#define CUDA_MEMCHECK_TK_VERSION_9_1       45
#define CUDA_MEMCHECK_TK_VERSION_10_0      CUDA_MEMCHECK_TK_VERSION_BUILD

/* The toolkit version which will act as feature gates */
/* The new output format was introduced in build 13 */
#define CUDA_MEMCHECK_TK_VERSION_NEWFORMAT 13
/* The backtrace section type was added to the output */
#define CUDA_MEMCHECK_TK_VERSION_BACKTRACE 14
/* The environment variable CUDA_MEMCHECK_OUTPUT format switched
   to version 2 (see above for the variable's format)
*/
#define CUDA_MEMCHECK_TK_VERSION_OUTPUT_V2 18

/* IPC capability added
*/
#define CUDA_MEMCHECK_TK_VERSION_IPC       25

/* IPC reply capability added
*/
#define CUDA_MEMCHECK_TK_VERSION_IPC_REPLY 31

/* Filter capability added */
#define CUDA_MEMCHECK_TK_VERSION_ADD_FILTER 36

/* UVM Fatal Faults added */
#define CUDA_MEMCHECK_TK_VERSION_UVM_FATAL_FAULTS 41

/* 64bit PCs added */
#define CUDA_MEMCHECK_TK_VERSION_64BIT_PCS 43

/* Unused Memory warning added */
#define CUDA_MEMCHECK_TK_VERSION_UNUSED_MEMORY 44

/* Deprecated instructions check flag added */
#define CUDA_MEMCHECK_TK_VERSION_DEPRECATED_INSTR 45

/* CUDA_MEMCHECK_FLAGS is the sum of the used options below. */
typedef enum {
    /* Detected violations fail the launch */
    CU_MEMCHECK_FLAG_FORCE_ERROR    = (1 << 0),
    /* Machine readable error record */
    CU_MEMCHECK_FLAG_RAW_OUTPUT     = (1 << 1),
    /* Invoked from the debugger (internal) */
    CU_MEMCHECK_FLAG_DEBUGGER       = (1 << 2),
    /* Detect and report hardware errors
      (incompatible with CU_MEMCHECK_FLAG_DEBUGGER) */
    CU_MEMCHECK_FLAG_ERRORS         = (1 << 3),
    /* Make launches non blocking (Fermi and above) */
    CU_MEMCHECK_FLAG_NONBLOCKING    = (1 << 4),
    /* Request dynamic memcheck to be enabled
       (requires Fermi, module using malloc())*/
    CU_MEMCHECK_FLAG_DYNAMIC        = (1 << 5),
    /* Enable race checking */
    CU_RACECHECK_FLAG_ENABLE        = (1 << 6),
    /* Enable API checks */
    CU_MEMCHECK_FLAG_API_CHECK      = (1 << 7),
    /* Force flushing to disk */
    CU_MEMCHECK_FLAG_FLUSH_TO_DISK  = (1 << 8),
    /* Host side backtrace */
    CU_MEMCHECK_FLAG_BACKTRACE_HOST = (1 << 9),
    /* Device side backtrace */
    CU_MEMCHECK_FLAG_BACKTRACE_DEV  = (1 << 10),
    /* cudaMalloc'd Global leak */
    CU_MEMCHECK_FLAG_GLOBAL_LEAK    = (1 << 11),
    /* Device heap leaks */
    CU_MEMCHECK_FLAG_DEVHEAP_LEAK   = (1 << 12),
    /* Enable racecheck analysis */
    CU_RACECHECK_FLAG_ANALYSIS      = (1 << 13),
    /* Set if this should suppress hazard */
    CU_RACECHECK_FLAG_ANALYSIS_ONLY = (1 << 14),
    /* Set if memcheck should not generate patches */
    CU_MEMCHECK_FLAG_DISABLE_PATCHING = (1 << 15),
    /* Set if memcpy/memset checking should be enabled */
    CU_MEMCHECK_FLAG_CHECK_MEMACCESS   = (1 << 16),
    /* Set if memcheck should warn on FatBinaryLoad */
    CU_MEMCHECK_FLAG_REPORT_FATBIN_LOAD_ERROR = (1 << 17),
    /* Enable Barcheck */
    CU_BARCHECK_FLAG_ENABLE         = (1 << 18),
    /* Enable Initcheck */
    CU_INITCHECK_FLAG_ENABLE        = (1 << 19),
    /* Enable UVM support */
    CU_MEMCHECK_FLAG_UVM            = (1 << 20),
    /* Enable unused memory records */
    CU_INITCHECK_FLAG_UNUSED_MEMORY = (1 << 21),
    /* Enable checking of deprecated instructions */
    CU_MEMCHECK_FLAG_CHECK_DEPRECATED = (1 << 22),
} CUmemcheck_flags;

/* Types of IPC reply records */
typedef enum {
    CU_MEMCHECK_IPC_REPLY_INVALID = 0,
    CU_MEMCHECK_IPC_REPLY_PID     = 1,

    /* Last entry */
    CU_MEMCHECK_IPC_REPLY_SIZE,

    CU_MEMCHECK_IPC_REPLY_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_ipc_replyType;

/* Types of IPC messages */
typedef enum {
    CU_MEMCHECK_IPC_MSG_INVALID    = 0,
    CU_MEMCHECK_IPC_MSG_INIT       = 1,
    CU_MEMCHECK_IPC_MSG_INIT_ACK   = 2, /* No payload */
    CU_MEMCHECK_IPC_MSG_INIT_NAK   = 3, /* No payload */
    CU_MEMCHECK_IPC_MSG_SET_OPTION = 4,
    CU_MEMCHECK_IPC_MSG_RUN_ASYNC  = 5, /* No payload */
    CU_MEMCHECK_IPC_MSG_GET_PID    = 6,
    CU_MEMCHECK_IPC_MSG_REPLY_DATA = 7,
    CU_MEMCHECK_IPC_MSG_ADD_FILTER = 8,
    CU_MEMCHECK_IPC_MSG_UVM_INFO   = 9,

    /* Last entry */
    CU_MEMCHECK_IPC_MSG_SIZE,

    CU_MEMCHECK_IPC_MSG_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_ipc_msgType;

/* Classes of records */
typedef enum {
    CU_MEMCHECK_RECORD_CLASS_INVALID = 0,
    CU_MEMCHECK_RECORD_CLASS_ERROR   = 1,
    CU_MEMCHECK_RECORD_CLASS_IPC     = 2,

    /* Last entry */
    CU_MEMCHECK_RECORD_CLASS_SIZE,

    CU_MEMCHECK_RECORD_CLASS_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_record_class;

/* Types of error records */
typedef enum {
    CU_MEMCHECK_RECORD_INVALID       = 0,
    CU_MEMCHECK_RECORD_MEM_PRECISE   = 1,
    CU_MEMCHECK_RECORD_MEM_IMPRECISE = 2,
    CU_MEMCHECK_RECORD_MEM_HINT      = 3, /* Deprecated */
    CU_MEMCHECK_RECORD_MEM_LEAK      = 4,
    CU_MEMCHECK_RECORD_DRV_ERROR     = 5,
    CU_MEMCHECK_RECORD_SHMEM_HAZARD  = 6,
    CU_MEMCHECK_RECORD_API_ERROR     = 7,
    CU_MEMCHECK_RECORD_MEM_SYSCALL   = 8,
    CU_MEMCHECK_RECORD_RACE_ANALYSIS = 9,
    CU_MEMCHECK_RECORD_MEM_ACCESS    = 10,
    CU_MEMCHECK_RECORD_BAR_VIOLATION = 11,
    CU_MEMCHECK_RECORD_MEM_INIT      = 12,
    CU_MEMCHECK_RECORD_UVM_FATAL_FAULT = 13,
    CU_MEMCHECK_RECORD_MEM_UNUSED    = 14,
    CU_MEMCHECK_RECORD_DRV_WARNING   = 15,

    /* Add new record types before this line. Also
       update their version in the array below */
    CU_MEMCHECK_RECORD_SIZE,

    CU_MEMCHECK_RECORD_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_record_types;

/* Flags set for leaks */
typedef enum  {
    CU_MEMCHECK_ALLOC_INTERNAL = 0x1,    /* Allocation is internal to driver */
    CU_MEMCHECK_ALLOC_NAMED    = 0x2,    /* Allocation has a name */
    CU_MEMCHECK_ALLOC_DEVHEAP  = 0x4,    /* Allocation is in device heap */
    CU_MEMCHECK_ALLOC_MODULE   = 0x8,    /* Allocation is module scoped */

    CU_MEMCHECK_ALLOC_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_alloc_flags;

/* Types of driver errors */
typedef enum {
    CU_MEMCHECK_ERROR_NONE                     = 0,
    CU_MEMCHECK_ERROR_UNKNOWN                  = 1,
    CU_MEMCHECK_ERROR_LAUNCH_SIZE_TOO_LARGE    = 2,
    CU_MEMCHECK_ERROR_UNSUPPORTED_DEVICE       = 3,
    CU_MEMCHECK_ERROR_OUT_OF_HOST_MEMORY       = 4,
    CU_MEMCHECK_ERROR_OUT_OF_DEVICE_MEMORY     = 5,
    CU_MEMCHECK_ERROR_FAILED_INITIALIZATION    = 6,
    CU_MEMCHECK_ERROR_DEVICE_MEM_ACCESS_FAILED = 7,
    CU_MEMCHECK_ERROR_FAILED_CLEANUP           = 8,
    CU_MEMCHECK_ERROR_DYNAMIC_INIT_FAILED      = 9,
    CU_MEMCHECK_ERROR_DYNAMIC_CLEANUP_FAILED   = 10,
    CU_MEMCHECK_ERROR_PROFILER_ATTACHED        = 11,
    CU_MEMCHECK_ERROR_UNSUPPORTED_OS           = 12,
    CU_MEMCHECK_ERROR_TOOL_ATTACHED            = 13,
    CU_MEMCHECK_ERROR_UNSUPPORTED_CNP          = 14,

    CU_MEMCHECK_ERROR_SIZE,

    CU_MEMCHECK_ERROR_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_error_types;

/* Levels for error records */
typedef enum {
    CU_MEMCHECK_RECORD_LEVEL_NONE   = 0,
    CU_MEMCHECK_RECORD_LEVEL_TRACE  = 10,
    CU_MEMCHECK_RECORD_LEVEL_INFO   = 20,
    CU_MEMCHECK_RECORD_LEVEL_WARN   = 30,
    CU_MEMCHECK_RECORD_LEVEL_ERROR  = 40,
    CU_MEMCHECK_RECORD_LEVEL_FATAL  = 50,

    CU_MEMCHECK_RECORD_LEVEL_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_record_levels;

/* Types of hazards */
typedef enum {
    CU_RACECHECK_HAZARD_NONE = 0,
    CU_RACECHECK_HAZARD_WAW  = 1,
    CU_RACECHECK_HAZARD_RAW  = 2,
    CU_RACECHECK_HAZARD_WAR  = 3,

    CU_RACECHECK_HAZARD_FORCE_SIZE = 0xFFFFFFFFU,
} CUracecheck_hazard_types;

/* Filter types for hazards */
typedef enum {
    CU_RACECHECK_FILTER_NONE = 0,
    CU_RACECHECK_FILTER_SAME_THREAD = (1 << 0),
    CU_RACECHECK_FILTER_SAME_DATA   = (1 << 1),
    CU_RACECHECK_FILTER_WARP_PROG   = (1 << 2),

    CU_RACECHECK_FILTER_FORCE_SIZE = 0xFFFFFFFFU,
} CUracecheck_hazard_filter_types;

/* Types of precise errors */
typedef enum {
    CU_MEMCHECK_ADDR_TYPE_NONE   = 0,
    CU_MEMCHECK_ADDR_TYPE_GLOBAL = 1,
    CU_MEMCHECK_ADDR_TYPE_LOCAL  = 2,
    CU_MEMCHECK_ADDR_TYPE_SHARED = 3,

    CU_MEMCHECK_ADDR_TYPE_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_addrType_types;

/* Syscall errors */
typedef enum {
    CU_MEMCHECK_SYSCALL_ERROR_NONE = 0,
    CU_MEMCHECK_SYSCALL_ERROR_INVALID_FREE = 1,
    CU_MEMCHECK_SYSCALL_ERROR_DOUBLE_FREE = 2,
    CU_MEMCHECK_SYSCALL_ERROR_HEAP_CORRUPT = 3,

    CU_MEMCHECK_SYSCALL_ERROR_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_syscallError_types;

/* API types */
typedef enum {
    CU_MEMCHECK_API_TYPE_NONE    = 0,
    CU_MEMCHECK_API_TYPE_DRIVER  = 1,
    CU_MEMCHECK_API_TYPE_RUNTIME = 2,

    CU_MEMCHECK_API_TYPE_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_apiType_types;

/* Memory access types */
typedef enum {
    CU_MEMCHECK_MEM_ACCESS_UNKNOWN      = 0,
    CU_MEMCHECK_MEM_ACCESS_MEMCPY_SRC   = 1,
    CU_MEMCHECK_MEM_ACCESS_MEMCPY_DST   = 2,
    CU_MEMCHECK_MEM_ACCESS_MEMSET_DST   = 3,

    CU_MEMCHECK_MEM_ACCESS_FORCE_SIZE   = 0xFFFFFFFFU,
} CUmemcheck_memAccess_types;

/* Memory access error types */
typedef enum {
    CU_MEMCHECK_MEM_ACCESS_ERROR_NONE = 0,
    CU_MEMCHECK_MEM_ACCESS_ERROR_UNKNOWN = 1,
    CU_MEMCHECK_MEM_ACCESS_ERROR_INVALID_RANGE  = 2,
    CU_MEMCHECK_MEM_ACCESS_ERROR_INVALID_START  = 3,
    CU_MEMCHECK_MEM_ACCESS_ERROR_INVALID_END    = 4,
    CU_MEMCHECK_MEM_ACCESS_ERROR_INTERNAL_ALLOC = 5,
    CU_MEMCHECK_MEM_ACCESS_ERROR_HEAP_CORRUPTION = 6,
    CU_MEMCHECK_MEM_ACCESS_ERROR_UNINITIALIZED  = 7,

    CU_MEMCHECK_MEM_ACCESS_ERROR_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_memAccessError_types;

/* Unused memory error types */
typedef enum {
    CU_MEMCHECK_MEM_UNUSED_ERROR_NONE         = 0,
    CU_MEMCHECK_MEM_UNUSED_ERROR_UNKNOWN      = 1,
    CU_MEMCHECK_MEM_UNUSED_ERROR_NOT_WRITTEN  = 2
} CUmemcheck_memUnusedError_types;

/* Barrier violation types */
typedef enum {
    CU_BARCHECK_BAR_VIOLATION_NONE                      = 0,
    CU_BARCHECK_BAR_VIOLATION_INTRA_WARP_DIVERGENCE     = 1,
    CU_BARCHECK_BAR_VIOLATION_STACK_DIVERGENCE          = 2,
    CU_BARCHECK_BAR_VIOLATION_INSUFFICIENT_LIVE_THREADS = 3,
    CU_BARCHECK_BAR_VIOLATION_UNKNOWN                   = 4,
    CU_BARCHECK_BAR_VIOLATION_INVALID_ARGUMENTS         = 5,
    CU_BARCHECK_BAR_VIOLATION_DEPRECATED_INSTRUCTION    = 6,

    CU_BARCHECK_BAR_VIOLATION_FORCE_SIZE = 0xFFFFFFFFU,
} CUbarcheck_barViolation_types;

/* Initcheck fields */
typedef enum {
    CU_INITCHECK_ACCESS_NONE             = 0,
    CU_INITCHECK_ACCESS_GLOBAL_LOAD      = 1,
    CU_INITCHECK_ACCESS_GLOBAL_ATOMIC    = 2,
    CU_INITCHECK_ACCESS_GLOBAL_REDUCTION = 3,

    CU_INITCHECK_ACCESS_FORCE_SIZE = 0xFFFFFFFFU,
} CUinitcheck_access_types;

/* Filter flags */
typedef enum {
    CU_MEMCHECK_FILTER_FLAG_NONE            = 0,
    CU_MEMCHECK_FILTER_FLAG_KERNEL_NAME_SUBSTRING  = (1 << 0),
    CU_MEMCHECK_FILTER_FLAG_KERNEL_NAME_EXACT      = (1 << 1),

    CU_MEMCHECK_FILTER_FLAG_FRCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_filterFlag_types;

/* Payload Record types */
typedef enum {
    CU_MEMCHECK_BACKTRACE_TYPE_INVALID = 0,
    CU_MEMCHECK_BACKTRACE_TYPE_HOST    = 1,
    CU_MEMCHECK_BACKTRACE_TYPE_DEVICE  = 2,

    CU_MEMCHECK_BACKTRACE_TYPE_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_backtrace_types;

typedef enum {
    CU_MEMCHECK_BACKTRACE_FRAME_FLAGS_NONE          = 0,
    CU_MEMCHECK_BACKTRACE_FRAME_FLAGS_HIDDEN        = (1 << 0),

    CU_MEMCHECK_BACKTRACE_FRAME_FLAGS_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_backtrace_frame_flags;

typedef enum {
    CU_MEMCHECK_UVM_VERSION_UNKNOWN  = 0,
    CU_MEMCHECK_UVM_VERSION_UVM8     = 1,
    CU_MEMCHECK_UVM_VERSION_UVM_LITE = 2,

    CU_MEMCHECK_UVM_VERSION_FORCE_SIZE = 0xFFFFFFFFU,
} CUmemcheck_uvm_version;

typedef struct CUmemcheck_error_record_st CUmemcheck_error_record;
typedef struct CUmemcheck_ipc_record_st CUmemcheck_ipc_record;
typedef struct CUmemcheck_record_string_st CUmemcheck_record_string;
typedef struct CUmemcheck_record_st CUmemcheck_record;

typedef struct CUmemcheck_record_type_memPrecise_st CUmemcheck_record_type_memPrecise;
typedef struct CUmemcheck_record_type_memImprecise_st CUmemcheck_record_type_memImprecise;
typedef struct CUmemcheck_record_type_memLeak_st CUmemcheck_record_type_memLeak;
typedef struct CUmemcheck_record_type_memSyscall_st CUmemcheck_record_type_memSyscall;
typedef struct CUmemcheck_record_type_driverError_st CUmemcheck_record_type_driverError;
typedef struct CUmemcheck_record_type_shmemHazard_st CUmemcheck_record_type_shmemHazard;
typedef struct CUmemcheck_record_type_hazardThreadInfo_st CUmemcheck_record_type_hazardThreadInfo;
typedef struct CUmemcheck_record_type_hazardCommonInfo_st CUmemcheck_record_type_hazardCommonInfo;
typedef struct CUmemcheck_record_type_apiError_st CUmemcheck_record_type_apiError;
typedef struct CUmemcheck_record_type_raceAnalysisVertex_st CUmemcheck_record_type_raceAnalysisVertex;
typedef struct CUmemcheck_record_type_raceAnalysisCommon_st CUmemcheck_record_type_raceAnalysisCommon;
typedef struct CUmemcheck_record_type_raceAnalysis_st CUmemcheck_record_type_raceAnalysis;
typedef struct CUmemcheck_record_type_memAccess_st CUmemcheck_record_type_memAccess;
typedef struct CUmemcheck_record_type_memUnused_st CUmemcheck_record_type_memUnused;
typedef struct CUmemcheck_record_type_barViolation_st CUmemcheck_record_type_barViolation;
typedef struct CUmemcheck_record_type_memInit_st CUmemcheck_record_type_memInit;
typedef struct CUmemcheck_record_type_uvmFatalFault_st CUmemcheck_record_type_uvmFatalFault;

typedef struct CUmemcheck_backtrace_st CUmemcheck_backtrace;
typedef struct CUmemcheck_backtrace_host_st CUmemcheck_backtrace_host;
typedef struct CUmemcheck_backtrace_dev_st CUmemcheck_backtrace_dev;
typedef struct CUmemcheck_backtrace_frame_host_st CUmemcheck_backtrace_frame_host;
typedef struct CUmemcheck_backtrace_frame_dev_st CUmemcheck_backtrace_frame_dev;

typedef struct CCipcmsg_type_replyDataPid_st CCipcmsg_type_replyDataPid;

typedef struct CCipcmsg_type_init_st CCipcmsg_type_init;
typedef struct CCipcmsg_type_setOption_st CCipcmsg_type_setOption;
typedef struct CCipcmsg_type_replyData_st CCipcmsg_type_replyData;
typedef struct CCipcmsg_type_addFilter_st CCipcmsg_type_addFilter;
typedef struct CCipcmsg_type_uvmInfo_st CCipcmsg_type_uvmInfo;

/*  Error type struct definitions
    Each struct corresponds to a particular error type.
 */
struct CUmemcheck_record_type_memPrecise_st {
    uint64_t address;
    uint32_t pc;             /* ~0 signifies unknown */
    uint32_t size;           /* Transfer size in bytes, 0 if unknown */
    uint32_t write;
    uint32_t blockIdxX, blockIdxY, blockIdxZ;
    uint32_t threadIdxX, threadIdxY, threadIdxZ;
    uint32_t lineno;        /* 0 if unknown */
    uint32_t type;
    CUDBGException_t exception;
    uint32_t funcNameIx;
    uint32_t fileNameIx;
};

struct CUmemcheck_record_type_memImprecise_st {
    uint64_t address;
    uint32_t pc;             /* ~0 signifies unknown */
    uint32_t blockIdxX, blockIdxY, blockIdxZ;
    uint32_t threadIdxX, threadIdxY, threadIdxZ;
    uint32_t lineno;
    CUDBGException_t exception;
    uint32_t funcNameIx;
    uint32_t fileNameIx;
};

struct CUmemcheck_record_type_memLeak_st {
    uint64_t size;          /* Size of leak */
    uint64_t address;       /* Device address of leak */
    uint32_t flags;         /* flags for allocation. Internal Use*/
    uint32_t nameIx;        /* Name for a named allocation */
};

struct CUmemcheck_record_type_driverError_st {
    uint32_t error;         /* The type of the error */
    uint32_t internal;      /* Boolean flag, 1 means internal */
    uint32_t stringIx;      /* Index of error string */
};

struct CUmemcheck_record_type_hazardThreadInfo_st {
    uint32_t write;            /* Set to true for write */
    uint32_t threadIdxX, threadIdxY, threadIdxZ; /* ThreadId */
    uint32_t pc;               /* PC of the original access */
    uint32_t line;             /* Line number of the source file */
    uint32_t fileNameIx;       /* File name for source file */
    uint32_t funcNameIx;       /* Function name */
    uint64_t rawPC;            /* NOT WRITTEN TO DISK. This is only valid for the in memory version */
};

struct CUmemcheck_record_type_hazardCommonInfo_st {
    uint64_t gridid;        /* Grid id, used by frontend to do clustering */
    uint32_t hazardType;    /* Type of hazard (CUracecheck_hazard_types) */
    uint32_t address;       /* Address in shared memory  */
    uint32_t blockIdxX, blockIdxY,blockIdxZ;
    uint32_t readData;      /* This is the value at the shared memory address */
    uint32_t writeData;     /* The value that was in flight at the last write */
    uint32_t filterType;    /* The actual filter condition used */
    uint32_t kernelNameIx;  /* Function name for kernel  */
};

struct CUmemcheck_record_type_shmemHazard_st {
    CUmemcheck_record_type_hazardCommonInfo common;
    CUmemcheck_record_type_hazardThreadInfo thread0;
    CUmemcheck_record_type_hazardThreadInfo thread1;
};

struct CUmemcheck_record_type_apiError_st {
    uint32_t apiType;
    uint32_t apiResult;
    uint64_t contextID;
    uint64_t apiCallID;
    uint32_t stringIx;
    uint32_t errorNameStringIx;
    uint32_t errorMsgStringIx;
};

struct CUmemcheck_record_type_memSyscall_st {
    uint32_t type;  /* Type of syscall error (CUmemcheck_syscallError_types) */
    uint32_t blockIdxX, blockIdxY, blockIdxZ;
    uint32_t threadIdxX, threadIdxY, threadIdxZ;
    uint32_t pc;        /* Currently will be the PC inside the syscall */
    uint32_t lineno;    /* Line in the code corresponding to the PC */
    uint64_t address;   /* Pointer value */
    uint64_t size;      /* Length of allocation */
    uint32_t fileNameIx;
    uint32_t funcNameIx;
};

struct CUmemcheck_record_type_raceAnalysisVertex_st {
    uint32_t pc;            /* Offset in the function */
    uint32_t write;         /* Whether the access is a write */
    uint32_t funcNameIx;    /* Name of function */
    uint32_t fileNameIx;    /* File name if available */
    uint32_t line;          /* Line number if available */
    uint32_t hazardTypeMask;    /* Only valid for a dest : type of hazard */
    uint32_t weight;            /* Only valid for a dest : Weight of the hazard */
    uint32_t count;             /* Only valid for a dest : number of hazards to dest */
};

struct CUmemcheck_record_type_raceAnalysisCommon_st {
    uint32_t numDests;
};

struct CUmemcheck_record_type_raceAnalysis_st {
    CUmemcheck_record_type_raceAnalysisCommon common;
    CUmemcheck_record_type_raceAnalysisVertex source;
    CUmemcheck_record_type_raceAnalysisVertex *dests;
};

struct CUmemcheck_record_type_memAccess_st {
    uint32_t accessType;    /* Type of access (CUmemcheck_memAccess_types) */
    uint32_t errorType;     /* Type of error (CUmemcheck_memAccessError_types) */
    uint64_t accessAddress; /* Address of attempted access*/
    uint64_t accessSize;    /* Size of access  */
    uint64_t errorAddress;  /* Address of first bad access */
};

struct CUmemcheck_record_type_memUnused_st {
    uint32_t errorType;     /* Type of error (CUmemcheck_memUnusedError_types) */
    uint64_t allocAddress;  /* Address of allocated block */
    uint64_t allocSize;     /* Size of allocated block */
    uint64_t unusedSize;    /* Amount of unused data */
    uint64_t blockAddress;  /* Address of page in the block, relative to allocAddress */
    uint64_t blockSize;     /* Size of block */
};

struct CUmemcheck_record_type_barViolation_st {
    uint32_t type;          /*  Type of violation (CUbarcheck_barViolation_type) */
    uint32_t barIndex;      /* Index of barrier */
    uint32_t pc;            /* PC of location (i.e. of barrier instruction) */
    uint32_t funcNameIx;    /* Function name */
    uint32_t fileNameIx;    /* File name */
    uint32_t line;          /* Line number */
    uint32_t blockIdxX, blockIdxY, blockIdxZ;
    uint32_t threadIdxX, threadIdxY, threadIdxZ;
};

struct CUmemcheck_record_type_memInit_st {
    uint64_t address;
    uint32_t pc;             /* ~0 signifies unknown */
    uint32_t size;           /* Transfer size in bytes, 0 if unknown */
    uint32_t type;
    uint32_t blockIdxX, blockIdxY, blockIdxZ;
    uint32_t threadIdxX, threadIdxY, threadIdxZ;
    uint32_t lineno;        /* 0 if unknown */
    uint32_t accessType;
    uint32_t funcNameIx;
    uint32_t fileNameIx;
};

struct CUmemcheck_record_type_uvmFatalFault_st {
    CUDBGUvmFaultType_t        faultType;
    CUDBGUvmMemoryAccessType_t accessType;
    CUDBGUvmFatalReason_t      reason;
    uint32_t                   processorIndex;
    uint64_t                   address;
    uint64_t                   timeStamp;
};

/* Error string helper */
struct CUmemcheck_record_string_st {
    char* str;
    uint32_t sz;
    uint32_t flags;
    uint32_t index;
    CUmemcheck_record_string *next;
};

struct CUmemcheck_backtrace_host_st {
    uint64_t threadId;
};

struct CUmemcheck_backtrace_dev_st {
    uint32_t threadIdxX;
    uint32_t threadIdxY;
    uint32_t threadIdxZ;
    uint32_t blockIdxX;
    uint32_t blockIdxY;
    uint32_t blockIdxZ;
    uint32_t entryKernelNameIx;
};

struct CUmemcheck_backtrace_frame_host_st {
    uint32_t flags;         /* Of frameFlags type */
    uint32_t depth;         /* Depth of this frame */
    uint64_t addr;          /* Return address of the frame */
    uint32_t funcNameIx;    /* Index for function name string */
    uint64_t funcOffset;    /* Offset of function */
    uint32_t libNameIx;     /* Index of library name string  */
    uint64_t libBaseAddr;   /* Base address of library */
    uint32_t fileNameIx;    /* File name (debug_info) */
    uint32_t lineno;        /* Line number (debug_info) */
};

struct CUmemcheck_backtrace_frame_dev_st {
    uint32_t flags;         /* Of frameFlags type */
    uint32_t depth;         /* Depth of this frame */
    uint64_t addr;          /* Return address of this frame */
    uint32_t funcNameIx;    /* Index of function name string */
    uint64_t funcOffset;    /* Offset of function */
    uint32_t kernelNameIx;  /* Index of kernel name string */
    uint64_t kernelAddr;    /* Base address of kernel */
    uint32_t moduleNameIx;  /* Index of module name string */
    uint32_t fileNameIx;    /* Index of file name string (debug info) */
    uint32_t lineno;        /* Line number in file (debug info) */
};

struct CUmemcheck_backtrace_st {
    uint32_t hostDepth;
    uint32_t maxHostDepth;

    uint32_t devDepth;
    uint32_t maxDevDepth;

    CUmemcheck_backtrace_host host;
    CUmemcheck_backtrace_dev  dev;

    CUmemcheck_backtrace_frame_host *hostFrames;
    CUmemcheck_backtrace_frame_dev  *devFrames;
};

/* CCipcmsg_type_replyDataPid
    Payload for the PID reply (a case of replyData message)
 */
struct CCipcmsg_type_replyDataPid_st {
    /* Record version parsed by backend */
    uint32_t recordVersion;
    /* Process ID */
    uint32_t pid;
};

/* CCipcmsg_type_init
    Payload for the init message
*/
struct CCipcmsg_type_init_st {
    /* Version of the handshake */
    uint32_t handshakeVersion;
    /* Minimum supported toolkit version */
    uint32_t minVersion;
    /* Maximum supported toolkit version */
    uint32_t maxVersion;
};

/* CCipcmsg_type_setOption
    Payload for the setOption message
 */
struct CCipcmsg_type_setOption_st {
    /* Record version parsed by backend */
    uint32_t recordVersion;
    /* Set option carries the raw option flag over */
    uint32_t flags;
};

/* CCipcmsg_type_replyData
    Payload for the replyData message
 */
struct CCipcmsg_type_replyData_st {
    CUmemcheck_ipc_replyType type;

    union {
        CCipcmsg_type_replyDataPid pid;
    } cases;
};

/* CCipcmsg_type_addFilter_st
   Filter payload
*/
struct CCipcmsg_type_addFilter_st {
    uint32_t flags; /* Mask of CUmemcheck_filterFlag_types */
    uint32_t entryNameIx;
};

/* CCipcmsg_type_uvmInfo_st
   Payload for the uvmInfo message
*/
struct CCipcmsg_type_uvmInfo_st {
    /* UVM version */
    CUmemcheck_uvm_version version;
    /* Number of handles sent after this message */
    uint32_t numHandles;
};

/* CUmemcheck_ipc_record_st
    For IPC messages
*/
struct CUmemcheck_ipc_record_st {
    CUmemcheck_ipc_msgType type;

    union {
        CCipcmsg_type_init init;
        CCipcmsg_type_setOption setOption;
        CCipcmsg_type_replyData replyData;
        CCipcmsg_type_addFilter addFilter;
        CCipcmsg_type_uvmInfo   uvmInfo;
    } cases;
};


/* CUmemcheck_error_record_st
    This is the internal representation of error records. These are converted
    to and from the disk record types. While individual cases are packed,
    the entire structure may not be.
 */
struct CUmemcheck_error_record_st {
    uint32_t    errortype;
    union {
        CUmemcheck_record_type_memPrecise       memPrecise;
        CUmemcheck_record_type_memImprecise     memImprecise;
        CUmemcheck_record_type_memLeak          memLeak;
        CUmemcheck_record_type_driverError      driverError;
        CUmemcheck_record_type_shmemHazard      shmemHazard;
        CUmemcheck_record_type_apiError         apiError;
        CUmemcheck_record_type_memSyscall       memSyscall;
        CUmemcheck_record_type_raceAnalysis     raceAnalysis;
        CUmemcheck_record_type_memAccess        memAccess;
        CUmemcheck_record_type_barViolation     barViolation;
        CUmemcheck_record_type_memInit          memInit;
        CUmemcheck_record_type_uvmFatalFault    uvmFatalFault;
        CUmemcheck_record_type_memUnused        memUnused;
    } cases;
};

/* CUmemcheck_record_st
    Internal representation of an arbitrary record (IPC and classical error)
*/
struct CUmemcheck_record_st {
    uint32_t    numchildren;
    CUmemcheck_record_levels level;
    CUmemcheck_record_class  recordclass;

    union {
        CUmemcheck_error_record error;
        CUmemcheck_ipc_record   ipc;
    } cases;

    /* All records form a doubly linked list */
    CUmemcheck_record         *next;
    CUmemcheck_record         *prev;

    /* Parent and child records */
    CUmemcheck_record         *parent;
    CUmemcheck_record         **children;

    /* String tracking structure */
    void                            *strings;

    /* Backtrace tracker */
    CUmemcheck_backtrace            *backtrace;
};



#endif
