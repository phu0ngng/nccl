/**
 * @file port_failure_emu_example.c
 * @brief Simple example demonstrating basic usage of the Port Failure Emulation library
 * 
 * This example shows how to:
 * 1. Initialize the library with device configuration
 * 2. Activate port failure emulation
 * 3. Perform operations while emulation is active
 * 4. Deactivate emulation and cleanup
 */

#include "port_failure_emu.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    // Initialize configuration
    const char *devices[] = {"mlx5_0", NULL};
    pfe_config_t config = {
        .device_names = devices,
        .num_devices = 1,
        .mode = PFE_MODE_RDMA
    };
    
    // Initialize library
    pfe_result_t result;
    pfe_context_t *ctx = pfe_init(&config, &result);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize Port Failure Emulation library (result: %d)\n", result);
        return 1;
    }
    
    printf("Library initialized successfully\n");
    
    // Activate port failure emulation on the first device
    const char *device = "mlx5_0";
    printf("Activating port failure emulation on %s...\n", device);
    
    result = pfe_activate(ctx, device);
    if (result != PFE_SUCCESS) {
        fprintf(stderr, "Failed to activate port failure emulation\n");
        pfe_destroy(ctx);
        return 1;
    }
    
    printf("Port failure emulation active. Simulating failure for 5 seconds...\n");
    sleep(5);
    
    // Deactivate emulation
    printf("Deactivating port failure emulation...\n");
    result = pfe_deactivate(ctx, device);
    if (result != PFE_SUCCESS) {
        fprintf(stderr, "Failed to deactivate port failure emulation\n");
    } else {
        printf("Port failure emulation deactivated\n");
    }
    
    // Cleanup
    pfe_destroy(ctx);
    printf("Cleanup complete\n");
    
    return 0;
}
