/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _INT_HANDLER_TEST_H_
#define _INT_HANDLER_TEST_H_

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

// {386FD24C-814A-4a44-A524-FE15220A0BF4}
CU_DEFINE_UUID(CU_ETID_intHandlerTest, 
    0x386fd24c, 0x814a, 0x4a44, 0xa5, 0x24, 0xfe, 0x15, 0x22, 0xa, 0xb, 0xf4);

// Define copies of the intHandler data types. These must be kept in lock-step
// with the actual types in intHandler.h.
typedef struct CUetblCuosEvent_st                   CUetblCuosEvent;
typedef struct CUetblIntHandler_st                  CUetblIntHandler;
typedef struct CUetblIntHandlerServiceRoutine_st    CUetblIntHandlerServiceRoutine;

typedef enum CUetblIntHandlerServiceFlags_enum
{
    CU_ETBL_INT_HANDLER_SERVICE_ANY_EVENT   = (1 << 0),
    CU_ETBL_INT_HANDLER_SERVICE_TIMEOUT     = (1 << 1)
} CUetblIntHandlerServiceFlags;

typedef enum CUetblIntHandlerServiceReason_enum
{
    CU_ETBL_INT_HANDLER_SERVICE_REASON_EVENT_SIGNAL = 0, // This routine's event was signaled
    CU_ETBL_INT_HANDLER_SERVICE_REASON_OTHER_EVENT  = 1, // Some other routine's event was signaled
    CU_ETBL_INT_HANDLER_SERVICE_REASON_TIMEOUT      = 2  // Timeout occured
} CUetblIntHandlerServiceReason;

typedef struct CUetblIntHandlerServiceRoutineParams_st
{
    CUetblIntHandlerServiceReason serviceReason;

    // Pointer to the data provided by the user at registerRoutine time.
    void *userData;
} CUetblIntHandlerServiceRoutineParams;


typedef struct CUetblIntHandlerTest_st
{
    // Need cuosEvent functionality (except for cuosEventWait)
    size_t (*cuosEventGetSize)(void);
    int (*cuosEventCreate)(CUetblCuosEvent *event);
    int (*cuosEventSignal)(CUetblCuosEvent *event);
    int (*cuosEventDestroy)(CUetblCuosEvent *event);

    CUresult (*intHandlerCreate)(CUetblIntHandler **intHandler);

    CUresult (*intHandlerDestroy)(CUetblIntHandler *intHandler);

    CUresult (*intHandlerRegisterRoutine)(CUetblIntHandler *                intHandler,
                                          CUetblIntHandlerServiceRoutine ** serviceRoutine,
                                          CUresult                          (*serviceRoutineCallback)(CUetblIntHandlerServiceRoutineParams *serviceParams),
                                          void *                            userData,
                                          CUetblCuosEvent *                 event,
                                          uint32_t                          serviceFlags,
                                          uint32_t                          registerFlags);

    CUresult (*intHandlerUnregisterRoutine)(CUetblIntHandler *              intHandler,
                                            CUetblIntHandlerServiceRoutine *serviceRoutine);

    CUresult (*intHandlerCheckError)(CUetblIntHandler *intHandler);
} CUetblIntHandlerTest;


#ifdef __cplusplus
} // extern "C"
#endif

#endif // _INT_HANDLER_TEST_H_
