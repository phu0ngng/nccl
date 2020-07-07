
#ifndef _WIZLET_H_
#define _WIZLET_H_

// #defines to make etbl look like normal driver API
// Look at //sw/gpgpu/cuda/import/cuda_etbl/memmap.h  for additional details
// on prototypes, data structures, enums, flag values, etc.
// shown here.
//
// Usage:
// #include "wizlet.h"
//
// some_early_func() {
//    init_etbl();
//    ...
// }
//
// other_func() {
//    ...
//    cuMemCreate(...);
//    ...
// }


#define CU_INIT_UUID_STATIC
#include <cuda_etbl/memmap.h>
#undef CU_INIT_UUID_STATIC

#define cuExperimentalMemAddressReserve etblMemmap->MemAddressReserve
#define cuExperimentalMemAddressFree etblMemmap->MemAddressFree
#define cuExperimentalMemMap etblMemmap->MemMap
#define cuExperimentalMemUnmap etblMemmap->MemUnmap
#define cuExperimentalMemSetAccess etblMemmap->MemSetAccess
#define cuExperimentalMemCreate etblMemmap->MemCreate
#define cuExperimentalMemRelease etblMemmap->MemRelease
#define cuExperimentalMemExportToShareableHandle etblMemmap->MemExportToShareableHandle
#define cuExperimentalMemImportFromShareableHandle etblMemmap->MemImportFromShareableHandle

static CUetblMemmap* etblMemmap = NULL;

// Get etbl
#define init_etbl() \
if (!etblMemmap) {\
    cuGetExportTable((const void**)&etblMemmap, &CU_ETID_Memmap);\
}

#endif // _WIZLET_H_
