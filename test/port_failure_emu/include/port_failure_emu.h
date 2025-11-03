/**
 * @file port_failure_emu.h
 * @brief Public API for Port Failure Emulation Library
 * 
 * This library facilitates port failure emulation for testing NCCL resiliency
 * features using DevX steering rules to drop RDMA packets at the NIC level.
 */

#ifndef PORT_FAILURE_EMU_H
#define PORT_FAILURE_EMU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/*                             Type Definitions                               */
/* ========================================================================== */

/**
 * @brief Opaque context handle for the port failure emulator
 * 
 * This handle is returned by pfe_init() and must be passed to all other
 * library functions. It encapsulates all internal state.
 */
typedef struct pfe_context pfe_context_t;

/**
 * @brief Result codes for library operations
 */
typedef enum {
    PFE_SUCCESS = 0,  /**< Operation completed successfully */
    PFE_ERROR = 1     /**< Operation failed */
} pfe_result_t;

/**
 * @brief Flow table creation mode
 */
typedef enum {
    PFE_MODE_RDMA,    /**< Create flow tables only for RDMA traffic (TX_RDMA/RX_RDMA) */
    PFE_MODE_ALL      /**< Create flow tables for all traffic (TX/RX and TX_RDMA/RX_RDMA) */
} pfe_mode_t;

/**
 * @brief Configuration structure for initializing the port failure emulator
 */
typedef struct {
    const char **device_names;  /**< Array of InfiniBand device names (e.g., "mlx5_0") */
    int num_devices;            /**< Number of devices in the array */
    pfe_mode_t mode;            /**< Flow table creation mode */
} pfe_config_t;

/**
 * @brief Status information for a single device
 */
typedef struct {
    const char *device_name;    /**< Name of the device */
    bool is_active;             /**< Whether port failure emulation is active */
} pfe_device_status_t;

/* ========================================================================== */
/*                          Lifecycle Functions                               */
/* ========================================================================== */

/**
 * @brief Initialize the port failure emulator
 * 
 * Creates a new context and prepares the specified devices for port failure
 * emulation. This function must be called before any other library functions.
 * 
 * @param config Configuration structure with device list and settings
 * @param result Output parameter for result code (can be NULL)
 * @return Context handle on success, NULL on failure
 * 
 * @note The caller must call pfe_destroy() to free resources when done
 */
pfe_context_t* pfe_init(const pfe_config_t *config, pfe_result_t *result);

/**
 * @brief Destroy the port failure emulator context and free all resources
 * 
 * Deactivates all active emulations and cleans up all internal resources.
 * After calling this function, the context handle becomes invalid.
 * 
 * @param ctx Context handle returned by pfe_init()
 */
void pfe_destroy(pfe_context_t *ctx);

/* ========================================================================== */
/*                        Activation/Deactivation                             */
/* ========================================================================== */

/**
 * @brief Activate port failure emulation on a specific device
 * 
 * Creates flow table entries to drop packets on the specified device.
 * The device must have been initialized in pfe_init().
 * 
 * @param ctx Context handle
 * @param device_name Name of the device (e.g., "mlx5_0")
 * @return PFE_SUCCESS on success, PFE_ERROR on failure
 */
pfe_result_t pfe_activate(pfe_context_t *ctx, const char *device_name);

/**
 * @brief Deactivate port failure emulation on a specific device
 * 
 * Removes flow table entries to restore normal packet flow.
 * 
 * @param ctx Context handle
 * @param device_name Name of the device (e.g., "mlx5_0")
 * @return PFE_SUCCESS on success, PFE_ERROR on failure
 */
pfe_result_t pfe_deactivate(pfe_context_t *ctx, const char *device_name);

/* ========================================================================== */
/*                          Status and Query                                  */
/* ========================================================================== */

/**
 * @brief Check if port failure emulation is active on a device
 * 
 * @param ctx Context handle
 * @param device_name Name of the device
 * @return true if emulation is active, false otherwise
 */
bool pfe_is_active(pfe_context_t *ctx, const char *device_name);

/**
 * @brief Get status information for all devices
 * 
 * Returns an array of status structures for all initialized devices.
 * The array is owned by the library and remains valid until pfe_destroy().
 * 
 * @param ctx Context handle
 * @param status Output pointer to status array
 * @param num_devices Output pointer for number of devices
 * @return PFE_SUCCESS on success, PFE_ERROR on failure
 */
pfe_result_t pfe_get_status(pfe_context_t *ctx, 
                            pfe_device_status_t **status, 
                            int *num_devices);

/* ========================================================================== */
/*                         Device Enumeration                                 */
/* ========================================================================== */

/**
 * @brief Get a list of all available InfiniBand devices on the system
 * 
 * Returns an array of device name strings. The caller must free the array
 * using pfe_free_device_list() when done.
 * 
 * @param device_names Output pointer to device name array
 * @param num_devices Output pointer for number of devices
 * @return PFE_SUCCESS on success, PFE_ERROR on failure
 */
pfe_result_t pfe_list_devices(char ***device_names, int *num_devices);

/**
 * @brief Free the device list returned by pfe_list_devices()
 * 
 * @param device_names Device name array to free
 * @param num_devices Number of devices in the array
 */
void pfe_free_device_list(char **device_names, int num_devices);

#ifdef __cplusplus
}
#endif

#endif /* PORT_FAILURE_EMU_H */