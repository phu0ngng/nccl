/**
 * @file pfe_core.c
 * @brief Core implementation of the Port Failure Emulation Library
 *
 * Implements context lifecycle, device enumeration, and configuration functions.
 */

#include "include/pfe_internal.h"
#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/*                          Helper Functions                                  */
/* ========================================================================== */

/**
 * @brief Get IB device by name from device list
 */
struct ibv_device* pfe_get_ib_dev_by_name(struct ibv_device **ibv_list,
                                           int nd,
                                           const char *name) {
    for (int i = 0; i < nd; i++) {
        if (strcmp(ibv_get_device_name(ibv_list[i]), name) == 0) {
            return ibv_list[i];
        }
    }
    return NULL;
}

/**
 * @brief Update the status cache for all devices
 */
void pfe_update_status_cache(pfe_context_t *ctx) {
    if (!ctx) return;

    for (int i = 0; i < ctx->num_devices; i++) {
        struct pfe_device_ctx *dev = &ctx->devices[i];
        pfe_device_status_t *status = &ctx->status_cache[i];

        status->device_name = dev->device_name;
        status->is_active = dev->is_active;
    }
}

/* ========================================================================== */
/*                          Lifecycle Functions                               */
/* ========================================================================== */

pfe_context_t* pfe_init(const pfe_config_t *config, pfe_result_t *result) {
    pfe_context_t *ctx = NULL;
    struct ibv_device **ibv_list = NULL;
    int ibv_dev_num = 0;
    pfe_result_t local_result = PFE_ERROR;

    // Validate input
    if (!config || !config->device_names || config->num_devices <= 0 ||
        config->num_devices > PFE_MAX_NUM_DEVICE) {
        WARN("Invalid configuration");
        goto error;
    }

    // Allocate context
    ctx = (pfe_context_t*)calloc(1, sizeof(pfe_context_t));
    if (!ctx) {
        WARN("Failed to allocate context");
        goto error;
    }

    ctx->mode = config->mode;
    ctx->num_devices = 0;

    // Get IB device list
    ibv_list = ibv_get_device_list(&ibv_dev_num);
    if (!ibv_list) {
        WARN("ibv_get_device_list failed");
        goto error;
    }

    INFO("Found %d InfiniBand devices", ibv_dev_num);

    // Initialize each requested device
    for (int i = 0; i < config->num_devices; i++) {
        const char *dev_name = config->device_names[i];
        struct ibv_device *ibv_dev = pfe_get_ib_dev_by_name(ibv_list, ibv_dev_num, dev_name);

        if (!ibv_dev) {
            WARN("Device not found: %s", dev_name);
            goto error;
        }

        INFO("Initializing device: %s", dev_name);

        // Initialize device context
        if (pfe_init_device(&ctx->devices[ctx->num_devices], ibv_dev,
                            ctx->mode) != PFE_INTERNAL_SUCCESS) {
            WARN("Failed to initialize device: %s", dev_name);
            goto error;
        }

        ctx->num_devices++;
    }

    // Update status cache
    pfe_update_status_cache(ctx);

    INFO("Successfully initialized %d devices", ctx->num_devices);
    local_result = PFE_SUCCESS;

    // Free device list (we've stored pointers to devices we need)
    ibv_free_device_list(ibv_list);

    if (result) *result = local_result;
    return ctx;

error:
    if (ibv_list) {
        ibv_free_device_list(ibv_list);
    }
    if (ctx) {
        pfe_destroy(ctx);
    }
    if (result) *result = PFE_ERROR;
    return NULL;
}

void pfe_destroy(pfe_context_t *ctx) {
    if (!ctx) return;

    INFO("Destroying context with %d devices", ctx->num_devices);

    // Deactivate and destroy all devices
    for (int i = 0; i < ctx->num_devices; i++) {
        struct pfe_device_ctx *dev = &ctx->devices[i];
        if (dev->is_initialized) {
            TRACE("Destroying device: %s", dev->device_name);

            // Deactivate if active
            if (dev->is_active) {
                pfe_deactivate_device(dev, ctx->mode);
            }

            // Destroy resources
            pfe_destroy_device_resources(dev, ctx->mode);

            // Close device context
            if (dev->ibv_ctx) {
                ibv_close_device(dev->ibv_ctx);
                dev->ibv_ctx = NULL;
            }
        }
    }

    free(ctx);
    INFO("Context destroyed");
}

/* ========================================================================== */
/*                         Device Enumeration                                 */
/* ========================================================================== */

pfe_result_t pfe_list_devices(char ***device_names, int *num_devices) {
    struct ibv_device **ibv_list = NULL;
    int ibv_dev_num = 0;
    char **names = NULL;

    if (!device_names || !num_devices) {
        return PFE_ERROR;
    }

    // Get IB device list
    ibv_list = ibv_get_device_list(&ibv_dev_num);
    if (!ibv_list || ibv_dev_num == 0) {
        *device_names = NULL;
        *num_devices = 0;
        return PFE_SUCCESS;  // No devices is not an error
    }

    // Allocate array for device names
    names = (char**)malloc(ibv_dev_num * sizeof(char*));
    if (!names) {
        ibv_free_device_list(ibv_list);
        return PFE_ERROR;
    }

    // Copy device names
    for (int i = 0; i < ibv_dev_num; i++) {
        const char *name = ibv_get_device_name(ibv_list[i]);
        names[i] = strdup(name);
        if (!names[i]) {
            // Cleanup on error
            for (int j = 0; j < i; j++) {
                free(names[j]);
            }
            free(names);
            ibv_free_device_list(ibv_list);
            return PFE_ERROR;
        }
    }

    ibv_free_device_list(ibv_list);

    *device_names = names;
    *num_devices = ibv_dev_num;
    return PFE_SUCCESS;
}

void pfe_free_device_list(char **device_names, int num_devices) {
    if (!device_names) return;

    for (int i = 0; i < num_devices; i++) {
        free(device_names[i]);
    }
    free(device_names);
}

/* ========================================================================== */
/*                          Status and Query                                  */
/* ========================================================================== */

pfe_result_t pfe_get_status(pfe_context_t *ctx,
                            pfe_device_status_t **status,
                            int *num_devices) {
    if (!ctx || !status || !num_devices) {
        return PFE_ERROR;
    }

    // Update cache before returning
    pfe_update_status_cache(ctx);

    *status = ctx->status_cache;
    *num_devices = ctx->num_devices;
    return PFE_SUCCESS;
}


