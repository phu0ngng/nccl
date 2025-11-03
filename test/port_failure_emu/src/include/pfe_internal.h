/**
 * @file pfe_internal.h
 * @brief Internal definitions for Port Failure Emulation Library
 * 
 * This file contains internal structures and function declarations that are
 * not exposed to library users. Only include this in library implementation files.
 */

#ifndef PFE_INTERNAL_H
#define PFE_INTERNAL_H

#include "../../include/port_failure_emu.h"
#include "mlx5_ifc.h"
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

/* ========================================================================== */
/*                             Internal Constants                             */
/* ========================================================================== */

/* Log levels for internal use */
typedef enum {
    PFE_LOG_WARN = 0,
    PFE_LOG_INFO = 1,
    PFE_LOG_TRACE = 2
} pfe_log_level_t;

#define PFE_MAX_NUM_DEVICE (8)
#define PFE_FT_LOG_SIZE (5)
#define PFE_FG_START_FLOW_IDX (0)
#define PFE_FG_END_FLOW_IDX (0xf)

#define ETHERTYPE_MASK (0xffff)
#define IP_PROTO_MASK (0xff)
#define UDP_DPORT_MASK (0xffff)
#define BTH_OPCODE_MASK (0xff)
#define BTH_DEST_QP_MASK (0xffffff)

#define IPv4_PROTOCOL (0x0800)
#define IPv6_PROTOCOL (0x86dd)
#define IP_PROTO_UDP (17)
#define ROCE_DPORT (4791)

/* ========================================================================== */
/*                              Logging Macros                                */
/* ========================================================================== */

static bool log_level_set = false;
static pfe_log_level_t config_level = PFE_LOG_WARN;

// Simple logging function (no context needed)
static inline void pfe_log(pfe_log_level_t level, const char *file, int line, 
                           const char *fmt, ...) {
    // Parse PFE_DEBUG environment variable to determine log level
    if (log_level_set == false) {
        const char* pfe_debug = getenv("PFE_DEBUG");
        if (pfe_debug == NULL) {
            config_level = PFE_LOG_INFO; // Default: show WARN and INFO, no TRACE
        } else {
            // Convert to uppercase for comparison
            char pfe_debug_upper[32];
            size_t len = strlen(pfe_debug);
            if (len >= sizeof(pfe_debug_upper)) {
                len = sizeof(pfe_debug_upper) - 1;
            }
            for (size_t i = 0; i < len; i++) {
                pfe_debug_upper[i] = toupper((unsigned char)pfe_debug[i]);
            }
            pfe_debug_upper[len] = '\0';
            
            // Parse log level
            if (strstr(pfe_debug_upper, "TRACE") != NULL) {
                config_level = PFE_LOG_TRACE;
            } else if (strstr(pfe_debug_upper, "INFO") != NULL) {
                config_level = PFE_LOG_INFO;
            } else if (strstr(pfe_debug_upper, "WARN") != NULL) {
                config_level = PFE_LOG_WARN;
            } else {
                // If PFE_DEBUG is set to any other value, enable WARN
                config_level = PFE_LOG_WARN;
            }
        }
        log_level_set = true;
    }
    
    // Filter messages based on configured level
    // WARN (0): Show only WARN messages
    // INFO (1): Show WARN and INFO messages
    // TRACE (2): Show WARN, INFO, and TRACE messages
    if (level > config_level) return;
    
    FILE* out = (level == PFE_LOG_WARN) ? stderr : stdout;
    const char* level_str;
    switch (level) {
        case PFE_LOG_WARN:  level_str = "WARN";  break;
        case PFE_LOG_INFO:  level_str = "INFO";  break;
        case PFE_LOG_TRACE: level_str = "TRACE"; break;
        default:            level_str = "???";   break;
    }
    
    fprintf(out, "[PFE-LIB %s %s:%d] ", level_str, file, line);
    
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    
    fprintf(out, "\n");
}

// Logging macros - Usage: INFO("Device %s activated", dev_name);
#define INFO(...)  pfe_log(PFE_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...)  pfe_log(PFE_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define TRACE(...) pfe_log(PFE_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)

/* ========================================================================== */
/*                         Internal Type Definitions                          */
/* ========================================================================== */

/**
 * @brief Internal result type for functions
 */
typedef enum { 
    PFE_INTERNAL_SUCCESS = 0, 
    PFE_INTERNAL_ERROR = 1 
} pfe_internal_result_t;

/**
 * @brief Resources for a single flow table (TX or RX)
 */
struct pfe_flow_table_resources {
    struct mlx5dv_devx_obj *flow_table;
    struct mlx5dv_devx_obj *flow_group;
    struct mlx5dv_devx_obj **flow_table_entries;
    uint32_t flow_table_id;
    uint32_t flow_group_id;
    int flow_table_entry_size;
};

/**
 * @brief Context for a single device
 */
struct pfe_device_ctx {
    struct ibv_device *ibv_dev;
    struct ibv_context *ibv_ctx;
    char device_name[64];
    bool is_active;
    bool is_initialized;

    // Resources for non-RDMA operations
    struct pfe_flow_table_resources tx;
    struct pfe_flow_table_resources rx;

    // Resources for RDMA operations
    struct pfe_flow_table_resources tx_rdma;
    struct pfe_flow_table_resources rx_rdma;
};

/**
 * @brief Main context structure (opaque to library users)
 */
struct pfe_context {
    struct pfe_device_ctx devices[PFE_MAX_NUM_DEVICE];
    int num_devices;
    pfe_mode_t mode;
    
    // Cached status array for pfe_get_status()
    pfe_device_status_t status_cache[PFE_MAX_NUM_DEVICE];
};

/* ========================================================================== */
/*                      Internal Function Declarations                        */
/* ========================================================================== */

/* ---------- Flow Table Operations (pfe_flow_table.c) ---------- */

/**
 * @brief Create a flow table on a device
 */
pfe_internal_result_t pfe_create_flow_table(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj **ft_obj,
                                             uint32_t *ft_id,
                                             uint32_t table_type,
                                             const char *desc);

/**
 * @brief Create a flow group on a device
 */
pfe_internal_result_t pfe_create_flow_group(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj **fg_obj,
                                             uint32_t *fg_id,
                                             uint32_t table_id,
                                             uint32_t table_type,
                                             const char *desc);

/**
 * @brief Create a flow table entry
 */
pfe_internal_result_t pfe_create_flow_entry(struct pfe_device_ctx *dev_ctx,
                                             struct mlx5dv_devx_obj ***fte_ptr,
                                             uint32_t fg_id,
                                             uint32_t ft_id,
                                             uint32_t table_type,
                                             int *fte_num,
                                             const char *desc);

/**
 * @brief Create all flow tables for a device (based on mode)
 */
pfe_internal_result_t pfe_create_device_flow_tables(struct pfe_device_ctx *dev_ctx,
                                                     pfe_mode_t mode);

/**
 * @brief Create all flow groups for a device (based on mode)
 */
pfe_internal_result_t pfe_create_device_flow_groups(struct pfe_device_ctx *dev_ctx,
                                                     pfe_mode_t mode);

/**
 * @brief Create all flow table entries for a device (based on mode)
 */
pfe_internal_result_t pfe_create_device_flow_entries(struct pfe_device_ctx *dev_ctx,
                                                      pfe_mode_t mode);

/**
 * @brief Destroy flow table resources
 */
void pfe_destroy_flow_table_resources(struct pfe_flow_table_resources *resources);

/**
 * @brief Destroy all device resources
 */
void pfe_destroy_device_resources(struct pfe_device_ctx *dev_ctx,
                                   pfe_mode_t mode);

/* ---------- Device Management (pfe_device.c) ---------- */

/**
 * @brief Find a device context by name
 */
struct pfe_device_ctx* pfe_find_device_by_name(pfe_context_t *ctx, const char *device_name);

/**
 * @brief Initialize a device context (open device, create flow tables/groups)
 */
pfe_internal_result_t pfe_init_device(struct pfe_device_ctx *dev_ctx,
                                       struct ibv_device *ibv_dev,
                                       pfe_mode_t mode);

/**
 * @brief Activate emulation on a device (create flow table entries)
 */
pfe_internal_result_t pfe_activate_device(struct pfe_device_ctx *dev_ctx,
                                           pfe_mode_t mode);

/**
 * @brief Deactivate emulation on a device (destroy flow table entries)
 */
pfe_internal_result_t pfe_deactivate_device(struct pfe_device_ctx *dev_ctx,
                                             pfe_mode_t mode);

/* ---------- Helper Functions ---------- */

/**
 * @brief Get IB device by name from device list
 */
struct ibv_device* pfe_get_ib_dev_by_name(struct ibv_device **ibv_list,
                                           int nd,
                                           const char *name);

/**
 * @brief Update status cache
 */
void pfe_update_status_cache(pfe_context_t *ctx);

/* ========================================================================== */
/*                              Helper Macros                                 */
/* ========================================================================== */

#define PFE_CHECK(call)                        \
  do {                                         \
    pfe_internal_result_t _status = (call);    \
    if (_status != PFE_INTERNAL_SUCCESS) {     \
      return _status;                          \
    }                                          \
  } while (0)

#endif /* PFE_INTERNAL_H */
