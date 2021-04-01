/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_cls_h__
#define __cuda_etbl_cls_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Context Local Storage interface for Cuda 3.1
//------------------------------------------------------------------

/* This provides an interface through which data may be attached to
 * and retrieved from a particular CUcontext.  In effect, this 
 * replaces the runtime's previous use of "thread-local storage" with
 * "context-local storage"
 *
 * This functions similarly to a "key -> data" map.  The effective key
 * to look up a piece of data comes in two parts, a CUcontext and a
 * key pointer.  The key pointer should be unique to each library
 * which will use this interface.  A robust way to do this (which the
 * runtime uses) is to have the key be the address of a global variable
 * in the library using this table.
 *
 * This provides a destruction callback mechanism whereby, in the event 
 * that a context is destroyed, all clients of this interface are 
 * informed of the tear-down so as they can tear-down any of their
 * internal data structures.  When this destruction call-back is triggered,
 * the context against which it was registered has already been destroyed
 * (so any CUDA API structures created against it are already invalid)
 *
 * This table is not to be versioned, but rather is to be replaced
 * wholesale should a different interface be needed due to radical
 * changes in the runtime's behavior.
 */

CU_DEFINE_UUID(CU_ETID_ContextLocalStorageInterface_v0301,
    0x6e3393c6, 0x2111, 0x11df, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93);

typedef struct CUetblContextLocalStorageInterface_v0301_st {
    /* Register a piece of data with a context.
     * - context is the context to associate the data with.  if NULL,
     *   use the context bound to the current thread.
     * - key a pointer which uniquely identifies the caller of this 
     *   interface.  it is suggested that this be a pointer to a global
     *   variable in the calling library.
     * - data is the data to associate with the context,key pair
     * - Destroy is a function to call in the event that context
     *   is destroyed (e.g, using cuCtxDestroy) before this 
     *   key is unregistered on context
     */
    CUresult (CUDAAPI *ContextLocalStorageEntryRegister)(
        CUcontext context,
        void *key,
        void *data,
        void (CUDAAPI *Destroy)(CUcontext context, void *key, void *data) );

    /* Unregister a key from a context
     * - context is the context that the data is associated with.  if NULL,
     *   use the context bound to the current thread.
     * - key is the key which was used to register some data.
     */
    CUresult (CUDAAPI *ContextLocalStorageEntryUnregister)(
        CUcontext context,
        void *key);
    
    /* Retrieve data from a context
     * - data will have the data registered with key in context returned in it
     * - context is the context against which data was registered with key.  if 
     *   NULL, use the context bound to the current thread.
     * - key is the key with which data was registered in context
     */
    CUresult (CUDAAPI *ContextLocalStorageEntryGet)(
        void **data,
        CUcontext context,
        void *key);

} CUetblContextLocalStorageInterface_v0301;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
