/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_trap_handler_h__
#define __cuda_etbl_tools_trap_handler_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "nvtypes.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// This table allows tools like debuggers to override trap handler behavior.
CU_DEFINE_UUID(CU_ETID_ToolsTrapHandler,
    0x949e52cc, 0x4e5c, 0x469d, 0x83, 0x7c, 0x96, 0x25, 0x86, 0x83, 0x34, 0xe4 );

typedef enum CUtoolsSyscallId_enum {
    CU_TOOLS_SYSCALL_ID_ANY       = 1, // return warps which have issued any syscall
    CU_TOOLS_SYSCALL_ID_ASSERT    = 2, // return warps which have issued assert
    CU_TOOLS_SYSCALL_ID_FORCE_INT = 0x3fffffff
} CUtoolsSyscallId;

typedef enum CUtoolsWarpTrapStatus_enum {
    CU_TOOLS_WARP_TRAP_STATUS_VALID_WARPS = 0,
    CU_TOOLS_WARP_TRAP_STATUS_TRAPPED_WARPS,
    CU_TOOLS_WARP_TRAP_STATUS_HANDLED_WARPS,
    CU_TOOLS_WARP_TRAP_STATUS_PAUSED_WARPS,
} CUtoolsWarpTrapStatus;

typedef enum CUtoolsTrapAckAction_enum {
    CU_TOOLS_TRAP_ACK_DEFAULT = 0,
    CU_TOOLS_TRAP_ACK_RESET,
    CU_TOOLS_TRAP_ACK_CONTINUE,
    CU_TOOLS_TRAP_ACK_PAUSE
} CUtoolsTrapAckAction;

// Parameter type for GetKeplerTrapHandlerInfo.
typedef struct CUtoolsGetKeplerTrapHandlerInfoParams_st {
    size_t struct_size;

    uint32_t knownWarpScratchPadVersion;
    uint32_t knownSmScratchPadVersion;
} CUtoolsGetKeplerTrapHandlerInfoParams;

// Contains info necessary to interface with driver's trap handler on Kepler arch.
typedef struct CUtoolsKeplerTrapHandlerInfo_st {
    size_t struct_size;

    // Debugger scratchPad memobj.
    // This is guaranteed to have a host pointer.
    CUtoolsMemObjHandle scratchPadMemObj;

    // Highest version of CUtoolsKeplerWarpScratchPad_v# this driver supports.
    // The version number = 1 for CUtoolsKeplerWarpScratchPad_v1.
    uint32_t warpScratchPadVersion;
    // Inter-warp stride for CUtoolsKeplerWarpScratchPad in the scratchPadMemObj.
    uint32_t warpScratchPadStride;
    // Offset of the warp scratch pad array in the scratchPadMemObj.
    uint32_t warpScratchPadOffset;
    // Offset of threadIdx for lane0 of the warp, from the start of CUtoolsKeplerWarpScratchPad.
    uint32_t warpThreadIdxOffset;

    // Highest version of CUtoolsKeplerSmScratchPad_v# this driver supports.
    uint32_t smScratchPadVersion;
    // Inter-SM stride for CUtoolsKeplerSmScratchPad in the scratchPadMemObj.
    uint32_t smScratchPadStride;
    // Offset of the SM scratch pad array in the scratchPadMemObj.
    uint32_t smScratchPadOffset;

    // Address of save-location for PR, in LMem.
    uint32_t prLMemAddr;

    // memobj for the debugger trap-handler instructions.
    CUtoolsMemObjHandle debuggerTrapHandler;
    // Offset within the debugger trap-handler for the INVOKE_AFTER_SAVE: NOP.
    size_t invokeAfterSaveOffset;
    // Offset within the debugger trap-handler for the INVOKE_BEFORE_RESTORE: NOP.
    size_t invokeBeforeRestoreOffset;
} CUtoolsKeplerTrapHandlerInfo;

typedef struct CUtoolsKeplerSmScratchPad_v1_st {
    uint64_t pauseServicedMask;
} CUtoolsKeplerSmScratchPad_v1;

typedef struct CUtoolsKeplerWarpScratchPad_v1_st {
    uint32_t virtID;               // SR3
    uint32_t ctaParam;             // QMD / c[0][0x48] CRS size per warp (pushed by method)
    uint32_t blockIdxX;            // SR37
    uint32_t blockIdxY;            // SR38
    uint32_t blockIdxZ;            // SR39
    uint32_t blockDimX;            // QMD / c[0][0x28]
    uint32_t blockDimY;            // QMD / c[0][0x2c]
    uint32_t blockDimZ;            // QMD / c[0][0x30]
    uint32_t gridId64Lo;           // QMD / c[0][0xdc] 
    uint32_t gridDimX;             // QMD / c[0][0x34]
    uint32_t gridDimY;             // QMD / c[0][0x38]
    uint32_t gridDimZ;             // QMD / c[0][0x3c]
    uint32_t shmemSizePrBlock;     // SR50
    uint32_t lmemWindowSize;       // SR53
    uint32_t lmemLoSizePrThread;   // SR54
    uint32_t lmemHiOff;            // SR55
    uint32_t globalErrorStatus;    // SR64
    uint32_t warpErrorStatus;      // SR66
    uint32_t warpBarrierState;     // This warp's barrier state
    uint32_t selfQMDLo;            // (Lo-bits) Pointer to the GPU QMD launch structure that belongs to this warp's grid
    uint32_t selfQMDHi;            // (Hi-bits) Pointer to the GPU QMD launch structure that belongs to this warp's grid
    uint32_t paramConstDptrLo;     // (Lo-bits) Pointer to the parameter bank of this warp's grid.
    uint32_t paramConstDptrHi;     // (Hi-bits) Pointer to the parameter bank of this warp's grid.
    uint32_t cirQueueIncrMinusOne; // SR41 - Increment of work (-1) associated with this CTA, also holds isQueue
    uint32_t lmemBaseLo;           // (Lo-bits) This warp's lmemBase (either default, or set via SETLMEMBASE)
    uint32_t lmemBaseHi;           // (Hi-bits) This warp's lmemBase (either default, or set via SETLMEMBASE)
    uint32_t crsPtr;               // The result of GETCRSPTR
} CUtoolsKeplerWarpScratchPad_v1;


// Parameter type for GetMaxwellTrapHandlerInfo.
typedef struct CUtoolsGetMaxwellTrapHandlerInfoParams_st {
    size_t struct_size;

    uint32_t knownSmScratchPadVersion;
} CUtoolsGetMaxwellTrapHandlerInfoParams;

// Contains info necessary to interface with driver's trap handler on Maxwell arch.
typedef struct CUtoolsMaxwellTrapHandlerInfo_st {
    size_t struct_size;

    // Debugger scratchPad memobj.
    // This is guaranteed to have a host pointer.
    CUtoolsMemObjHandle scratchPadMemObj;

    // Highest version of CUtoolsMaxwellSmScratchPad_v# this driver supports.
    uint32_t smScratchPadVersion;
    // Inter-SM stride for CUtoolsMaxwellSmScratchPad in the scratchPadMemObj.
    uint32_t smScratchPadStride;
    // Offset of the SM scratch pad array in the scratchPadMemObj.
    uint32_t smScratchPadOffset;

    // memobj for the debugger trap-handler instructions.
    CUtoolsMemObjHandle driverTrapHandler;
    // Offset within the driver trap-handler for the JCAL DEBUGGER_TRAP_HANDLER instruction.
    size_t jcalDebuggerOffset;
} CUtoolsMaxwellTrapHandlerInfo;

typedef struct CUtoolsMaxwellSmScratchPad_v1_st {
    uint64_t pauseServicedMask;
} CUtoolsMaxwellSmScratchPad_v1;


typedef struct CUetblToolsTrapHandler_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    // Register the existence of Parallel Nsight's trap handler.
    // This is a one-way switch; there is no way to unregister the Nsight trap handler.
    //
    // To avoid thread-safety issues, call this in the cuInit callback.
    CUresult (CUDAAPI *RegisterInprocTrapHandler)(void);

    // Checks if an in-process trap handler has been attached.
    NvBool (CUDAAPI *IsInprocTrapHandlerPresent)(void);

    // Functions for examining/issuing syscalls after a debug event and 
    // full suspension of the device.
    
    // The use of these functions should be vaguely similar to the 
    // following:
    //
    // (1) Debug event received by event handler or debugger 
    //     wishes to issue a device suspend.
    // (2) Event handler should suspend the device, wait for pause, 
    //     and read out the state of each sm (which warps have paused, 
    //     which warps have issued traps)
    // (3) Examine the state of each sm to determine if this debug 
    //     event was the result of an exception, breakpoint, or 
    //     syscall (using IdentifyWarpsInSyscall() to identify 
    //     which bpt.traps were for breakpoints or syscalls.
    // (4) Process the syscalls for the context if any syscalls were 
    //     issued (using ProcessSyscalls())
    // (5) If the debug event was the result of solely syscalls, 
    //     resume execution of the device. Otherwise remain paused 
    //     and continue as normal.

    // \brief Returns a bit mask of which warps have have trapped in order
    // to make a syscall on the specifed sm. This call must be made after
    // the requested sm is completely suspended. Nsight needs this call to
    // differentiate between bpt.traps issued for breakpoints and bpt.traps
    // issued to perform a syscall.
    //
    // \param ctx - context which owns the currently suspended device launch
    // \param syscallBitVector - returned bit vector of warps on the suspended 
    // sm which have issued the requested syscalls
    // \param smid - smid of sm which is to be examined
    // \param trapBitVector - bit vector of warps on the sm have issued traps. 
    // \param id - specifies which syscalls are being queried. Specifically, 
    // if we want a bit vector of warps which have issued any syscall, or just 
    // the warps which have issued assert. This is important because nsight 
    // should treat an assert syscall as an nsight initiated breakpoint, yet 
    // should continue execution after traps made for other syscalls.
    CUresult (CUDAAPI *IdentifyWarpsInSyscall)(CUcontext ctx,
                                               uint64_t *syscallBitVector,
                                               uint32_t smid,
                                               uint64_t trapBitVector, 
                                               CUtoolsSyscallId syscall);

    // \brief Process host side work for all warps on the device which have 
    // issued syscalls. This call must only be made once the entire device 
    // has been suspended. After this call has been made, issued syscall 
    // state can no longer be examined with IdentifyWarpsInSyscall().
    //
    // \param ctx - context which owns the currently suspended kernel launch
    // \param trapBitVector - bit vector array with one uint64_t per sm. Each 
    // sm's bit vector specifies the warps which have issued trap instructions,
    // and suspended. The bitmasks for each sm are ordered by smid - there 
    // must be a bitmask for each sm on the device.
    CUresult (CUDAAPI *ProcessSyscalls)(CUcontext ctx, 
                                        uint64_t *trapBitVector);

    // Sets the trap-handler address for the context via push-buffer.
    CUresult (CUDAAPI *SetTrapHandlerAddress)(
        CUcontext ctx,
        CUtoolsMemObjHandle hMemObj);

    // \brief Query the trap state warps on a given SM.  Results are given in
    // a uin64_t bit field, one bit for each warp.  Supports up to 64 warps
    // per SM, though this limit varies with hardware architecture.
    //
    // \param ctx - context which owns the trapped warps.
    // \param smid - ID of the SM to query.
    // \param statType - which status to query: valid warps, trapped warps,
    // paused warps, or handled warps.
    // \param result - pointer to where results should be copied.
    CUresult (CUDAAPI *GetSMWarpTrapStatus)(
        CUcontext ctx,
        uint32_t smid,
        CUtoolsWarpTrapStatus statType,
        uint64_t *result);

    // \brief Set traps as having been handled for warps on a given SM.  Callee
    // can only set bits.  They cannot unset bits.
    //
    // \param ctx - context which owns the trapped warps.
    // \param smid - ID of the SM to set handle warps for.
    // \param warps - Bits to set.
    CUresult (CUDAAPI *SetSMHandledWarps)(
        CUcontext ctx,
        uint32_t smid,
        uint64_t warps);

    // \brief Query to get current acknowledgement action determined by
    // trap handling processing.
    //
    // \param ctx - context to query.
    // \param ackAction - pointer to where query result is to be stored.
    CUresult (CUDAAPI *GetTrapAckAction)(
        CUcontext ctx,
        CUtoolsTrapAckAction *ackAction);

    // \brief Set the acknowledgement action to take in response to a trap.
    // Callee may override current setting (queried by GetTrapAckAction).
    // ** Action is ignored if GPU is in an unrecoverable state (sm error,
    //    unhandled traps, syscall assert).  These events always result
    //    in a reset unless they are dealt with directly (by a debugger). **
    //
    // \param ctx - context for whose channel is to be reset.
    // \param ackAction - the acknowledgement action to take.
    CUresult (CUDAAPI *SetTrapAckAction)(
        CUcontext ctx,
        CUtoolsTrapAckAction ackAction);

    // \brief Force the driver trap handler and debug event handler thread 
    // to be initialized on all contexts created in this process where it 
    // would otherwise not be initialized.
    CUresult (CUDAAPI *EnableTrapHandler)(uint32_t enable);

    // Get Kepler trap handler info.
    CUresult (CUDAAPI *GetKeplerTrapHandlerInfo)(
        CUcontext ctx,
        const CUtoolsGetKeplerTrapHandlerInfoParams *params,
        CUtoolsKeplerTrapHandlerInfo *info);

    // Get Maxwell trap handler info.
    CUresult (CUDAAPI *GetMaxwellTrapHandlerInfo)(
        CUcontext ctx,
        const CUtoolsGetMaxwellTrapHandlerInfoParams *params,
        CUtoolsMaxwellTrapHandlerInfo *info);
} CUetblToolsTrapHandler;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
