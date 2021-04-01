/* 
* Copyright 2009-2011 by NVIDIA Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to NVIDIA
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of NVIDIA Corporation is prohibited.
*/

/******************************************************************************
*
*   Module: apiprofclient.h
*
*   Description:
*       CUDA profiler client api specific defines, structures, routines etc
*
******************************************************************************/

#ifndef __API_PROF_CLIENT_H__
#define __API_PROF_CLIENT_H__

#include "nvtypes.h"
#include "limits.h"

// packed alignment defines for struct etc
#if defined(_WIN32) // Windows 32- and 64-bit
#define START_PACKED_ALIGNMENT __pragma(pack(push,1))      // exact fit - no padding
#define PACKED_ALIGNMENT    
#define END_PACKED_ALIGNMENT __pragma(pack(pop))           // back to whatever the previous packing mode was
#elif defined(__GNUC__) // GCC
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__ ((__packed__))
#define END_PACKED_ALIGNMENT
#else // all other compilers
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#define CUPROFCL_NUM_PARAMS                     4
#define CUPROFCL_MAX_MESSAGES                   8       // keep 2^x always

// defines for global profiling modes
#define CUPROFCL_PROF_MODE_AUTO_GLOBAL                                  1      // legacy mode
#define CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_ENABLED_AT_START     2      // manual start/stop, launch with profiling enabled
#define CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_DISABLED_AT_START    4      // manual start/stop, launch with profiling disabled

#define CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_MASK (CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_ENABLED_AT_START | CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_DISABLED_AT_START)
#define CUPROFCL_PROF_MODE_MASK (CUPROFCL_PROF_MODE_AUTO_GLOBAL | CUPROFCL_PROF_MODE_MANUAL_GLOBAL_PROFILING_MASK)

// error codes
typedef enum CUProfClStatus_enum {
    CUPROFCL_STATUS_SUCCESS                         = 0,
    CUPROFCL_STATUS_ERROR_IF_INIT                   = 1,
    CUPROFCL_STATUS_ERROR_INVALID_IF_HANDLE         = 2,
    CUPROFCL_STATUS_ERROR_DRIVER_INTERFACE          = 3,
    CUPROFCL_STATUS_ERROR_TIMEOUT                   = 4,
    CUPROFCL_STATUS_ERROR_IN_PROCESSING             = 5,
    CUPROFCL_STATUS_ERROR_OUT_OF_MEMORY             = 6,
    CUPROFCL_STATUS_ERROR_VERSION_MISMATCH          = 7,
    CUPROFCL_STATUS_ERROR_DATA_BUFFER_FULL          = 8,
    CUPROFCL_STATUS_ERROR_UNIMPLEMENTED             = 9,
    CUPROFCL_STATUS_ERROR_INVALID_HANDLE            = 10,
    CUPROFCL_STATUS_ERROR_INVALID_VALUE             = 11,
    CUPROFCL_STATUS_ERROR_UNKNOWN                   = 99,
    // define a enum value large enough to force the type to be at least 4 bytes as sizeof enum type is compiler specific 
    CUPROFCL_STATUS_FORCE_TO_INT                    = INT_MAX   
} CUProfClStatus;

// request status code
typedef enum CUProfClReqStatus_enum {
    CUPROFCL_REQ_STATUS_EMPTY                       = 0,
    CUPROFCL_REQ_STATUS_READY                       = 1,
    CUPROFCL_REQ_STATUS_PROCESSING                  = 2,
    CUPROFCL_REQ_STATUS_COMPLETED                   = 3,
    CUPROFCL_REQ_STATUS_FORCE_TO_INT                = INT_MAX
} CUProfClReqStatus;

// profiling request options
typedef enum CUProfClReqOptions_enum {
    CUPROFCL_SHM_REQ_UNDEFINED                      = -1,
    CUPROFCL_SHM_REQ_INIT                           = 1,
    CUPROFCL_SHM_REQ_SET_PROFILING                  = 2,
    CUPROFCL_SHM_REQ_FORCE_TO_INT                   = INT_MAX
} CUProfClReqOptions;

START_PACKED_ALIGNMENT
// shared memory header struct
typedef struct PACKED_ALIGNMENT CUProfClShmHeader_st {
    char signature[12];
    NvU32 version;
    NvU32 sizeHeader;
    NvU32 sizeData;
    NvU32 offsetData;
    volatile NvU32 writeCounter;                    // CUDA client(RW), driver(R)  - total messages written (no wrap around), not pointing to message slots
    volatile NvU32 readCounter;                     // CUDA client(R),  driver(RW) - total messages read (no wrap around), not pointing to message slots
} CUProfClShmHeader;

// message struct
typedef struct PACKED_ALIGNMENT CUProfClMessage_st {
    volatile CUProfClReqOptions clientRequestId;                        // CUDA client(W),  driver(R) - one of the supported profiler req option
    volatile NvS32 clientRequestParam[CUPROFCL_NUM_PARAMS];             // CUDA client(W),  driver(R) - based on the req option, driver interprets it
    volatile CUProfClReqStatus serverResponseData[CUPROFCL_NUM_PARAMS]; // CUDA client(RW), driver(W) - acknowledgement by server
    volatile CUProfClStatus status;                                     // CUDA client(R),  driver(W) - return code by server
} CUProfClMessage;
END_PACKED_ALIGNMENT

// various defines related to shared memory
#define CUPROFCL_SHM_NAME_PREFIX           "nvcuprofsmem"
#define CUPROFCL_SHM_HEADER_SIGNATURE      "NVCUPROF"
#define CUPROFCL_SHM_HEADER_VERSION        1
#define CUPROFCL_SHM_HEADER_SIZE           sizeof(CUProfClShmHeader)
#define CUPROFCL_SHM_DATA_SIZE             (sizeof(CUProfClMessage) * CUPROFCL_MAX_MESSAGES)
#define CUPROFCL_SHM_DATA_OFFSET           CUPROFCL_SHM_HEADER_SIZE
#define CUPROFCL_SHM_TOTAL_SIZE            (CUPROFCL_SHM_HEADER_SIZE + CUPROFCL_SHM_DATA_SIZE)

#endif // #ifndef __API_PROF_CLIENT_H__
