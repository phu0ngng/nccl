/**
 * @file pfe_device.c
 * @brief Device management for Port Failure Emulation Library
 * 
 * Implements per-device activation/deactivation and status query functions.
 */

#include "include/pfe_internal.h"
#include <string.h>

/* ========================================================================== */
/*                          Helper Functions                                  */
/* ========================================================================== */

struct pfe_device_ctx* pfe_find_device_by_name(pfe_context_t *ctx, const char *device_name) {
    if (!ctx || !device_name) return NULL;

    for (int i = 0; i < ctx->num_devices; i++) {
        if (strcmp(ctx->devices[i].device_name, device_name) == 0) {
            return &ctx->devices[i];
        }
    }
    return NULL;
}

/* ========================================================================== */
/*                          Device Initialization                             */
/* ========================================================================== */

pfe_internal_result_t pfe_init_device(struct pfe_device_ctx *dev_ctx,
                                       struct ibv_device *ibv_dev,
                                       pfe_mode_t mode) {
    if (!dev_ctx || !ibv_dev) {
        return PFE_INTERNAL_ERROR;
    }

    // Initialize device context structure
    memset(dev_ctx, 0, sizeof(*dev_ctx));
    dev_ctx->ibv_dev = ibv_dev;
    dev_ctx->is_active = false;
    dev_ctx->is_initialized = false;
    strncpy(dev_ctx->device_name, ibv_get_device_name(ibv_dev), 
            sizeof(dev_ctx->device_name) - 1);

    // Open device with DevX support
    dev_ctx->ibv_ctx = mlx5dv_open_device(ibv_dev, &(struct mlx5dv_context_attr){
        .flags = MLX5DV_CONTEXT_FLAGS_DEVX,
    });
    if (!dev_ctx->ibv_ctx) {
        WARN("Failed to open device %s with DevX support", dev_ctx->device_name);
        return PFE_INTERNAL_ERROR;
    }

    // Verify that the device is RoCE/Ethernet or exit otherwise
    // TODO: Check why devices configured with IB (not RoCE/Ethernet) are not
    // working and creation of a flow table returns a syndrome 0x2ac754.
    struct ibv_port_attr port_attr;
    if (ibv_query_port(dev_ctx->ibv_ctx, 1, &port_attr) != 0) {
        WARN("Failed to query port attributes for device %s", dev_ctx->device_name);
        goto error_close_device;
    }
    if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        WARN("Device %s is not RoCE/Ethernet (link layer: %d) and is not supported", dev_ctx->device_name, port_attr.link_layer);
        goto error_close_device;
    }

    // Create flow tables
    if (pfe_create_device_flow_tables(dev_ctx, mode) != PFE_INTERNAL_SUCCESS) {
        WARN("Failed to create flow tables for device %s", dev_ctx->device_name);
        goto error_close_device;
    }

    // Create flow groups
    if (pfe_create_device_flow_groups(dev_ctx, mode) != PFE_INTERNAL_SUCCESS) {
        WARN("Failed to create flow groups for device %s", dev_ctx->device_name);
        goto error_destroy_tables;
    }

    dev_ctx->is_initialized = true;
    return PFE_INTERNAL_SUCCESS;

error_destroy_tables:
    pfe_destroy_device_resources(dev_ctx, mode);

error_close_device:
    if (dev_ctx->ibv_ctx) {
        ibv_close_device(dev_ctx->ibv_ctx);
        dev_ctx->ibv_ctx = NULL;
    }
    return PFE_INTERNAL_ERROR;
}

/* ========================================================================== */
/*                      Device Activation/Deactivation                        */
/* ========================================================================== */

pfe_internal_result_t pfe_activate_device(struct pfe_device_ctx *dev_ctx,
                                           pfe_mode_t mode) {
    if (!dev_ctx || !dev_ctx->is_initialized) {
        return PFE_INTERNAL_ERROR;
    }

    if (dev_ctx->is_active) {
        return PFE_INTERNAL_SUCCESS;
    }

    // Create flow table entries to drop packets
    if (pfe_create_device_flow_entries(dev_ctx, mode) != PFE_INTERNAL_SUCCESS) {
        WARN("Failed to activate device %s", dev_ctx->device_name);
        return PFE_INTERNAL_ERROR;
    }

    dev_ctx->is_active = true;
    return PFE_INTERNAL_SUCCESS;
}

pfe_internal_result_t pfe_deactivate_device(struct pfe_device_ctx *dev_ctx,
                                             pfe_mode_t mode) {
    if (!dev_ctx || !dev_ctx->is_initialized) {
        return PFE_INTERNAL_ERROR;
    }

    if (!dev_ctx->is_active) {
        return PFE_INTERNAL_SUCCESS;
    }

    // Destroy flow table entries to restore normal packet flow
    // Destroy TX RDMA entries
    if (dev_ctx->tx_rdma.flow_table_entries) {
        for (int i = 0; i < dev_ctx->tx_rdma.flow_table_entry_size; i++) {
            if (dev_ctx->tx_rdma.flow_table_entries[i]) {
                mlx5dv_devx_obj_destroy(dev_ctx->tx_rdma.flow_table_entries[i]);
            }
        }
        free(dev_ctx->tx_rdma.flow_table_entries);
        dev_ctx->tx_rdma.flow_table_entries = NULL;
        dev_ctx->tx_rdma.flow_table_entry_size = 0;
    }

    // Destroy RX RDMA entries
    if (dev_ctx->rx_rdma.flow_table_entries) {
        for (int i = 0; i < dev_ctx->rx_rdma.flow_table_entry_size; i++) {
            if (dev_ctx->rx_rdma.flow_table_entries[i]) {
                mlx5dv_devx_obj_destroy(dev_ctx->rx_rdma.flow_table_entries[i]);
            }
        }
        free(dev_ctx->rx_rdma.flow_table_entries);
        dev_ctx->rx_rdma.flow_table_entries = NULL;
        dev_ctx->rx_rdma.flow_table_entry_size = 0;
    }

    // Destroy non-RDMA entries if mode is MODE_ALL
    if (mode == PFE_MODE_ALL) {
        // Destroy TX entries
        if (dev_ctx->tx.flow_table_entries) {
            for (int i = 0; i < dev_ctx->tx.flow_table_entry_size; i++) {
                if (dev_ctx->tx.flow_table_entries[i]) {
                    mlx5dv_devx_obj_destroy(dev_ctx->tx.flow_table_entries[i]);
                }
            }
            free(dev_ctx->tx.flow_table_entries);
            dev_ctx->tx.flow_table_entries = NULL;
            dev_ctx->tx.flow_table_entry_size = 0;
        }

        // Destroy RX entries
        if (dev_ctx->rx.flow_table_entries) {
            for (int i = 0; i < dev_ctx->rx.flow_table_entry_size; i++) {
                if (dev_ctx->rx.flow_table_entries[i]) {
                    mlx5dv_devx_obj_destroy(dev_ctx->rx.flow_table_entries[i]);
                }
            }
            free(dev_ctx->rx.flow_table_entries);
            dev_ctx->rx.flow_table_entries = NULL;
            dev_ctx->rx.flow_table_entry_size = 0;
        }
    }

    dev_ctx->is_active = false;
    return PFE_INTERNAL_SUCCESS;
}

/* ========================================================================== */
/*                      Public API Implementation                             */
/* ========================================================================== */

pfe_result_t pfe_activate(pfe_context_t *ctx, const char *device_name) {
    if (!ctx || !device_name) {
        WARN("Invalid arguments to pfe_activate");
        return PFE_ERROR;
    }

    struct pfe_device_ctx *dev = pfe_find_device_by_name(ctx, device_name);
    if (!dev) {
        WARN("Device not found: %s", device_name);
        return PFE_ERROR;
    }

    if (pfe_activate_device(dev, ctx->mode) != PFE_INTERNAL_SUCCESS) {
        return PFE_ERROR;
    }

    INFO("Port failure emulation activated on device %s", device_name);
    pfe_update_status_cache(ctx);
    return PFE_SUCCESS;
}

pfe_result_t pfe_deactivate(pfe_context_t *ctx, const char *device_name) {
    if (!ctx || !device_name) {
        WARN("Invalid arguments to pfe_deactivate");
        return PFE_ERROR;
    }

    struct pfe_device_ctx *dev = pfe_find_device_by_name(ctx, device_name);
    if (!dev) {
        WARN("Device not found: %s", device_name);
        return PFE_ERROR;
    }

    if (pfe_deactivate_device(dev, ctx->mode) != PFE_INTERNAL_SUCCESS) {
        return PFE_ERROR;
    }

    INFO("Port failure emulation deactivated on device %s", device_name);
    pfe_update_status_cache(ctx);
    return PFE_SUCCESS;
}

bool pfe_is_active(pfe_context_t *ctx, const char *device_name) {
    if (!ctx || !device_name) {
        return false;
    }

    struct pfe_device_ctx *dev = pfe_find_device_by_name(ctx, device_name);
    if (!dev) {
        return false;
    }

    return dev->is_active;
}
