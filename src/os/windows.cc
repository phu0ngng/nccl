#include <cstdint>

uint64_t ncclOsGetpid() {
    return (uint64_t)GetCurrentProcessId();
}
