/**
 * @file port_failure_emu_cli.c
 * @brief Command-line interface for Port Failure Emulation Library
 * 
 * This CLI tool provides a simple interface to activate port failure emulation
 * on InfiniBand devices for testing purposes.
 */

#include "port_failure_emu.h"
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================== */
/*                             Global State                                   */
/* ========================================================================== */

// Global variable to control the main loop
static volatile sig_atomic_t should_exit = 0;

// Global context for cleanup in signal handler
static pfe_context_t *g_ctx = NULL;

/* ========================================================================== */
/*                          Signal Handler                                    */
/* ========================================================================== */

static void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        printf("[PFE-CLI INFO %s:%d] Received signal %d, initiating cleanup and exit\n", __FILE__, __LINE__, sig);
        should_exit = 1;
    }
}

/* ========================================================================== */
/*                          Usage and Parsing                                 */
/* ========================================================================== */

static void print_usage(const char *program_name) {
    printf("Usage: %s -d deviceList [-m mode] [-v]\n", program_name);
    printf("  -d deviceList  Comma-separated list of devices (e.g., mlx5_0,mlx5_1)\n");
    printf("  -m mode        Mode for flow table creation: rdma or all (default: rdma)\n");
    printf("\n");
}

static void print_available_devices(void) {
    char **device_names = NULL;
    int num_devices = 0;

    if (pfe_list_devices(&device_names, &num_devices) == PFE_SUCCESS) {
        printf("Available devices:\n");
        for (int i = 0; i < num_devices; i++) {
            printf("  - %s\n", device_names[i]);
        }
        pfe_free_device_list(device_names, num_devices);
    } else {
        printf("Failed to enumerate devices\n");
    }
}

static int parse_device_list(const char *device_list_str, char ***devices, int *num_devices) {
    if (!device_list_str || !devices || !num_devices) {
        return -1;
    }

    // Count devices
    int count = 1;
    for (const char *p = device_list_str; *p; p++) {
        if (*p == ',') count++;
    }

    // Allocate array
    char **dev_array = (char**)malloc(count * sizeof(char*));
    if (!dev_array) {
        return -1;
    }

    // Parse comma-separated list
    char *list_copy = strdup(device_list_str);
    if (!list_copy) {
        free(dev_array);
        return -1;
    }

    char *token = strtok(list_copy, ",");
    int i = 0;
    while (token && i < count) {
        dev_array[i] = strdup(token);
        if (!dev_array[i]) {
            // Cleanup on error
            for (int j = 0; j < i; j++) {
                free(dev_array[j]);
            }
            free(dev_array);
            free(list_copy);
            return -1;
        }
        token = strtok(NULL, ",");
        i++;
    }

    free(list_copy);

    *devices = dev_array;
    *num_devices = i;
    return 0;
}

static void free_device_list(char **devices, int num_devices) {
    if (!devices) return;
    for (int i = 0; i < num_devices; i++) {
        free(devices[i]);
    }
    free(devices);
}

/* ========================================================================== */
/*                              Main Function                                 */
/* ========================================================================== */

int main(int argc, char *argv[]) {
    int opt;
    char *device_list_str = NULL;
    pfe_mode_t mode = PFE_MODE_RDMA;
    char **devices = NULL;
    int num_devices = 0;
    pfe_result_t result;
    int exit_code = 0;

    // Parse command-line arguments
    while ((opt = getopt(argc, argv, "d:m:vh")) != -1) {
        switch (opt) {
        case 'd':
            device_list_str = optarg;
            break;
        case 'm':
            if (strcmp(optarg, "rdma") == 0) {
                mode = PFE_MODE_RDMA;
            } else if (strcmp(optarg, "all") == 0) {
                mode = PFE_MODE_ALL;
            } else {
                fprintf(stderr, "[PFE-CLI ERROR %s:%d] Invalid mode '%s'. Use 'rdma' or 'all'.\n", __FILE__, __LINE__, optarg);
                print_usage(argv[0]);
                print_available_devices();
                return 1;
            }
            break;
        case 'h':
            print_usage(argv[0]);
            print_available_devices();
            return 0;
        default:
            print_usage(argv[0]);
            print_available_devices();
            return 1;
        }
    }

    // Validate required arguments
    if (!device_list_str) {
        fprintf(stderr, "[PFE-CLI ERROR %s:%d] -d (device list) argument is required.\n\n", __FILE__, __LINE__);
        print_usage(argv[0]);
        print_available_devices();
        return 1;
    }

    // Parse device list
    if (parse_device_list(device_list_str, &devices, &num_devices) != 0) {
        fprintf(stderr, "[PFE-CLI ERROR %s:%d] Failed to parse device list\n", __FILE__, __LINE__);
        return 1;
    }

    printf("[PFE-CLI INFO %s:%d] Port Failure Emulation CLI\n", __FILE__, __LINE__);
    printf("[PFE-CLI INFO %s:%d] Mode: %s\n", __FILE__, __LINE__, mode == PFE_MODE_RDMA ? "RDMA" : "ALL");
    printf("[PFE-CLI INFO %s:%d] Devices: ", __FILE__, __LINE__);
    for (int i = 0; i < num_devices; i++) {
        printf("%s%s", devices[i], i < num_devices - 1 ? ", " : "\n");
    }

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize library
    pfe_config_t config = {
        .device_names = (const char**)devices,
        .num_devices = num_devices,
        .mode = mode
    };

    g_ctx = pfe_init(&config, &result);
    if (!g_ctx) {
        fprintf(stderr, "[PFE-CLI ERROR %s:%d] Failed to initialize port failure emulator\n", __FILE__, __LINE__);
        free_device_list(devices, num_devices);
        return 1;
    }

    // Activate port failure emulation on all devices
    printf("[PFE-CLI INFO %s:%d] Activating port failure emulation on all devices...\n", __FILE__, __LINE__);
    for (int i = 0; i < num_devices; i++) {
        if (pfe_activate(g_ctx, devices[i]) != PFE_SUCCESS) {
            fprintf(stderr, "[PFE-CLI ERROR %s:%d] Failed to activate device %s\n", __FILE__, __LINE__, devices[i]);
            pfe_destroy(g_ctx);
            free_device_list(devices, num_devices);
            return 1;
        }
    }

    printf("[PFE-CLI INFO %s:%d] Port failure emulation activated on all devices\n", __FILE__, __LINE__);
    printf("[PFE-CLI INFO %s:%d] Press Ctrl+C (SIGINT) or send SIGTERM to terminate...\n", __FILE__, __LINE__);

    // Main loop - wait for signal
    while (!should_exit) {
        usleep(100000); // 100ms
    }

    // Cleanup
    printf("[PFE-CLI INFO %s:%d] Cleaning up...\n", __FILE__, __LINE__);
    for (int i = 0; i < num_devices; i++) {
        pfe_deactivate(g_ctx, devices[i]);
    }
    pfe_destroy(g_ctx);
    free_device_list(devices, num_devices);

    printf("[PFE-CLI INFO %s:%d] Cleanup complete. Exiting.\n", __FILE__, __LINE__);
    return exit_code;
}
