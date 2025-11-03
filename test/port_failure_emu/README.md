
# Port Failure Emulation (PFE) Library

This library facilitates port failure emulation for testing NCCL resiliency features. It provides both a **C library API** for programmatic use and a **CLI tool** for manual testing.

The library is implemented over DevX API that allows insertion of steering rules into the NIC that make the NIC drop packets, to simulate network failures.

## Features

- ✅ **Programmatic API**: Integrate port failure emulation into your test suites
- ✅ **Per-Device Control**: Activate/deactivate emulation on specific NICs.
- ✅ **Multiple Modes**: RDMA-only or all traffic.
- ✅ **Runtime Control**: Enable/disable emulation without restarting the NIC(s).
- ✅ **Status Queries**: Check emulation state.
- ✅ **CLI Tool**: Ready-to-use command-line interface.

## Prerequisites

The following packages must be installed on the system:

* `libmlx5-1` (for building)
* `libibverbs-dev` (for building)
* `pkg-config` (for building)
* `ninja-build` (for building, optional)
* `cmake` (for building, optional)

## Building

### Compilation Instructions

The library can be built using `make` or `cmake` with `ninja`. 

To use `make`, simply run from NCCL root directory:

```bash
make -C test/port_failure_emu

```
This will produce the shared library `libportfailureemu.so`, the CLI tool `portFailureEmu`, and the example program `portFailureEmuExample` in the `build` directory

Below are the steps for `cmake`/`ninja`:

```bash
sudo apt update
sudo apt install libmlx5-1 libibverbs-dev rdma-core ninja-build cmake -y
mkdir build
cd build
cmake ../ -GNinja
ninja
```

### Build Artifacts

After building, you'll find:

```
<BUILT_DIRECTORY>
├── libportfailureemu.so      # Shared library
├── portFailureEmu            # CLI tool
└── portFailureEmuExample     # Example program
```

## Usage

### As a Library

#### Basic Example

```c
#include <port_failure_emu.h>

int main() {
    // Configure the library
    const char *devices[] = {"mlx5_0"};
    pfe_config_t config = {
        .device_names = devices,
        .num_devices = 1,
        .mode = PFE_MODE_RDMA
    };

    // Initialize
    pfe_context_t *ctx = pfe_init(&config, NULL);
    
    // Activate port failure emulation
    pfe_activate(ctx, "mlx5_0");
    
    // Run your tests here...
    
    // Deactivate and cleanup
    pfe_deactivate(ctx, "mlx5_0");
    pfe_destroy(ctx);
    
    return 0;
}
```

Compile and link with:
```bash
gcc -o my_test my_test.c -lportfailureemu
```

#### Advanced Example: Per-Device Control

```c
#include <port_failure_emu.h>

int main() {
    const char *devices[] = {"mlx5_0", "mlx5_1"};
    pfe_config_t config = {
        .device_names = devices,
        .num_devices = 2,
        .mode = PFE_MODE_RDMA
    };

    pfe_context_t *ctx = pfe_init(&config, NULL);
    
    // Fail only mlx5_0
    pfe_activate(ctx, "mlx5_0");
    
    // Check status
    if (pfe_is_active(ctx, "mlx5_0")) {
        printf("mlx5_0 is in failure mode\n");
    }
    
    // Recover mlx5_0, fail mlx5_1 instead
    pfe_deactivate(ctx, "mlx5_0");
    pfe_activate(ctx, "mlx5_1");
    
    // Get detailed status
    pfe_device_status_t *status;
    int num_devices;
    pfe_get_status(ctx, &status, &num_devices);
    
    for (int i = 0; i < num_devices; i++) {
        printf("Device %s: %s\n", 
               status[i].device_name,
               status[i].is_active ? "FAILED" : "OK");
    }
    
    pfe_destroy(ctx);
    return 0;
}
```

### As a CLI Tool

The CLI tool provides a simple interface for manual testing:

```bash
./portFailureEmu -d <device_list> [-m mode]
```

**Options:**
- `-d device_list`: Comma-separated list of devices (e.g., `mlx5_0,mlx5_1`)
- `-m mode`: Flow table mode - `rdma` (default) or `all`
- `-h`: Show help and list available devices

**Examples:**

```bash
# Fail a single device (RDMA traffic only)
./portFailureEmu -d mlx5_0

# Fail multiple devices
./portFailureEmu -d mlx5_0,mlx5_1

# Fail all traffic (not just RDMA)
./portFailureEmu -d mlx5_0 -m all

# List available devices
./portFailureEmu -h
```

The tool will run until you press **Ctrl+C** or send **SIGTERM**.

### Running the Example

```bash
cd <BUILD_DIRECTORY>
./portFailureEmuExample
```

This example demonstrates:
1. Checking available devices
2. Initializing the library
3. Activating emulation
4. Checking status
5. Deactivating and cleanup

### Usage in NCCL

NCCL can use this library both as a library and via the CLI tool for resiliency tests. 

## API Reference

### Context Lifecycle

```c
pfe_context_t* pfe_init(const pfe_config_t *config, pfe_result_t *result);
void pfe_destroy(pfe_context_t *ctx);
```

### Activation/Deactivation

```c
pfe_result_t pfe_activate(pfe_context_t *ctx, const char *device_name);
pfe_result_t pfe_deactivate(pfe_context_t *ctx, const char *device_name);
```

### Status and Query

```c
bool pfe_is_active(pfe_context_t *ctx, const char *device_name);
pfe_result_t pfe_get_status(pfe_context_t *ctx, pfe_device_status_t **status, int *num_devices);
```

### Device Enumeration

```c
pfe_result_t pfe_list_devices(char ***device_names, int *num_devices);
void pfe_free_device_list(char **device_names, int num_devices);
```

**Note:** Mode (`PFE_MODE_RDMA` or `PFE_MODE_ALL`) is set during initialization via `pfe_config_t` and cannot be changed dynamically. To use a different mode, destroy the context and create a new one.

**Logging:** The library can optionally use NCCL's logger by providing a `ncclDebugLogger_t` callback in the config. When no logger is provided, WARN and INFO messages are printed to stderr/stdout. TRACE messages are only output when an NCCL logger is provided.

## Modes

### PFE_MODE_RDMA (Default)

Creates flow tables for RDMA traffic only:
- TX RDMA flow table
- RX RDMA flow table

Use this mode to specifically test RDMA communication failures.

### PFE_MODE_ALL

Creates flow tables for all traffic:
- TX RDMA flow table
- RX RDMA flow table
- TX flow table
- RX flow table

Use this mode to simulate complete port/NIC failure.

## Integration with NCCL Tests

Example of integrating with NCCL resiliency tests:

```c
#include "port_failure_emu.h"
#include "nccl.h"

void test_nccl_port_failure_recovery() {
    // Setup NCCL
    ncclComm_t comms[4];
    setup_nccl_comms(comms, 4);
    
    // Initialize port failure emulator
    const char *devices[] = {"mlx5_0"};
    pfe_config_t config = {
        .device_names = devices,
        .num_devices = 1,
        .mode = PFE_MODE_RDMA
    };
    pfe_context_t *pfe_ctx = pfe_init(&config, NULL);
    
    // Start NCCL operation
    start_nccl_allreduce(comms);
    
    // Inject failure
    usleep(100000);
    pfe_activate(pfe_ctx, "mlx5_0");
    
    // Wait for NCCL to detect and recover
    wait_for_recovery(comms);
    
    // Restore port
    pfe_deactivate(pfe_ctx, "mlx5_0");
    
    // Verify recovery
    verify_operations_work(comms);
    
    // Cleanup
    pfe_destroy(pfe_ctx);
    cleanup_nccl(comms);
}
```

## Custom `mlx5_ifc.h`

The port failure emulation library uses a customized version of the [`mlx5_ifc.h` header provided by rdma-core](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/mlx5_ifc.h).

The custom version exposes interfaces which are not present/exposed in upsteram version in rdma-core.

The specific interfaces used in this library are:

* `MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX_RDMA` and `MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX_RDMA`
    * These types are not available in the upstream version.
    * custom `mlx5_ifc.h`:
        ```c
        enum {
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX              = 0x0,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX              = 0x1,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_EGRESS_ACL  = 0x2,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_INGRESS_ACL = 0x3,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_FDB         = 0x4,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_SNIFFER_RX      = 0x5,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_SNIFFER_TX      = 0x6,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_RX_RDMA         = 0x7,  <--- Used in the library
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_NIC_TX_RDMA         = 0x8,  <--- Used in the library
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_PORT_SELECTION      = 0x9,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_FDB_RX      = 0xa,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_FDB_TX      = 0xb,
            MLX5_CREATE_FLOW_TABLE_IN_TABLE_TYPE_ESWITCH_FDB_UNIFIED = 0xc,
        };
        ```
    * upstream `mlx5_ifc.h`:
        ```c
        enum {
            FS_FT_NIC_RX          = 0x0,
            FS_FT_NIC_TX          = 0x1,
            FS_FT_ESW_EGRESS_ACL  = 0x2,
            FS_FT_ESW_INGRESS_ACL = 0x3,
            FS_FT_FDB             = 0X4,
            FS_FT_SNIFFER_RX      = 0X5,
            FS_FT_SNIFFER_TX      = 0X6,
        };
        ```
* `MLX5_CMD_OPCODE_SET_FLOW_TABLE_ENTRY`
    * In upstream version, it's called `MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY`
* `MLX5_SET_FLOW_TABLE_ENTRY_IN_OP_MOD_SET`
    * Is not available in the upstream version.
* `MLX5_FLOW_CONTEXT_ACTION_DROP`
    * In upsream version, the "Drop" (`0x2`) action is not available. The only available actions are:
        ```c
        enum {
            MLX5_FLOW_CONTEXT_ACTION_FWD_DEST = 0x4,
            MLX5_FLOW_CONTEXT_ACTION_COUNT    = 0x8,
        };
        ```
* `MLX5_CMD_OPCODE_CREATE_FLOW_TABLE`
    * In upstream version, it's called `MLX5_CMD_OP_CREATE_FLOW_TABLE`
* `MLX5_CMD_OPCODE_CREATE_FLOW_GROUP`
    * In upstream version, it's called `MLX5_CMD_OP_CREATE_FLOW_GROUP`
* `MLX5_CREATE_FLOW_GROUP_IN_GROUP_TYPE_TCAM_SUBTABLE`
    * Is not available in the upstream version.

## Troubleshooting

### "Device not found" error

- Ensure the device name is correct (check with `ibv_devices` for example or `ibstat`)
- Verify the device supports DevX

### "Failed to open device" error

- Check that you have appropriate permissions
- Ensure `rdma-core` and `libmlx5` are properly installed

### Compilation errors

- Verify all prerequisites are installed.
- Check that `libmlx5` and `libibverbs` are found.

## License

See LICENSE.txt for details.