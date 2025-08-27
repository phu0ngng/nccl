#include <cstdint>
#include <unistd.h>

// Process Management
uint64_t ncclOsGetpid() {
    return (uint64_t)getpid();
}
