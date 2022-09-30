#ifndef _ETBL_MULTICAST_
#define _ETBL_MULTICAST_

#define CU_INIT_UUID_STATIC
#include <cuda_etbl/multicast.h>
#undef CU_INIT_UUID_STATIC

#define pfn_cuMemDeviceSupportsMulticast etblMulticast->DeviceSupportsMulticast
#define pfn_cuMemMulticastCreate etblMulticast->MulticastCreate
#define pfn_cuMemMulticastBindMem etblMulticast->MulticastBindMem

static const CUetblMulticast *etblMulticast = NULL;

// Get etbl
#define init_etbl() \
  if (!etblMulticast) {							\
    CUCHECK(cuGetExportTable((const void **)&etblMulticast, &CU_ETID_Multicast)); \
    fprintf(stderr, "etblMulticast = %p MulticastCreate %p MulticastBindMem %p\n", etblMulticast, pfn_cuMemMulticastCreate, pfn_cuMemMulticastBindMem); \
    assert(etblMulticast != NULL); \
    assert(pfn_cuMemMulticastCreate != NULL); \
    assert(pfn_cuMemMulticastBindMem != NULL); \
}


#endif /* _ETBL_MULTICAST_ */
