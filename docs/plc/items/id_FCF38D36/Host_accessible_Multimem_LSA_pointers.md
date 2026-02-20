<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
This feature adds host-accessible functions to retrieve LSA (Local/Store Accessible) pointers from NCCL device runtime, enabling host-side access to peer and multimem pointers.
This has been requested because it simplifies shared kernel development between different frameworks.
The implementation provides three new API functions: `ncclGetLsaMultimemDevicePointer`, `ncclGetLsaDevicePointer`, and `ncclGetPeerDevicePointer` that allow host code to obtain pointers to LSA memory regions that were previously only accessible from device code.
These functions automatically discover the associated communicator from the window object, eliminating the need to pass the communicator explicitly.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
- https://nvbugspro.nvidia.com/bug/5466255
- https://jirasw.nvidia.com/browse/GP-417

### User Experience
- NCCL device API users can now access LSA pointers from host code
- Enables compatibility with existing custom communication kernels that expect raw pointers instead of NCCL window objects
- Simplifies integration by allowing direct pointer passing to kernels rather than requiring window object handling
- **Simplified API**: Functions no longer require communicator parameter - they automatically discover the communicator from the window object

### Assumptions, constraints and dependencies
- Requires existing LSA functionality to be properly initialized
- Depends on NCCL device runtime shadow pool mechanism and RAS list of communicators for communicator discovery
- Assumes valid NCCL window objects (communicator is discovered automatically)
- Requires proper error handling for invalid arguments
- Pointer lifetime is limited to the shorter of Window and Communicator lifetime
- Functions use global communicator registry for automatic communicator discovery

### Use Cases
- **Custom Device Kernels**: Existing custom communication kernels that expect raw pointers in their signatures can now be used directly with NCCL windows without modification
- **Legacy Code Integration**: Enables seamless integration of existing communication kernels that were designed for raw pointer access
- **Simplified API**: Reduces complexity for users who prefer direct pointer access over window object management

### Platform Requirements
- Compatible with existing NCCL LSA implementation
- No additional hardware requirements beyond LSA support

<!-- ### Functional Requirements -->
<!-- ### System Requirements -->
<!-- ### Interface Requirements -->
<!-- ### KPI Requirements -->
<!-- ### Security Requirements -->
<!-- ### Legal and Standards Requirements -->
<!-- ### Telemetry Requirements -->
<!-- ### Backward Compatibility Requirements -->
<!-- ### Virtualization Requirements -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

The design extends the existing NCCL device runtime with four new host-accessible functions:

1. **`ncclGetLsaMultimemDevicePointer`**: Retrieves the multimem base pointer
2. **`ncclGetMultimemDevicePointer`**: Retrieves the multimem base pointer using a provided multimem handle
3. **`ncclGetLsaDevicePointer`**: Retrieves the pointer to a specific LSA peer's memory region
4. **`ncclGetPeerDevicePointer`**: Retrieves the pointer to a specific world rank peer's memory region

All four functions follow the same pattern as existing device functions but operate on the host side, utilizing the shadow pool mechanism to convert device windows to host-accessible representations.
**Key improvement**: Functions automatically discover the associated communicator from the window object, eliminating the need for users to pass the communicator explicitly.

### Interface Architecture

The new functions are declared in `src/include/nccl_device/core.h` and implemented in `src/dev_runtime.cc`:

```c
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaMultimemDevicePointer(
    ncclWindow_t window,
    size_t offset,
    void** outPtr
);

NCCL_EXTERN_C __host__ ncclResult_t ncclGetMultimemDevicePointer(
    ncclWindow_t window,
    size_t offset,
    ncclMultimemHandle multimem,
    void** outPtr
);

NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaDevicePointer(
    ncclWindow_t window,
    size_t offset,
    int lsaRank,
    void** outPtr
);

NCCL_EXTERN_C __host__ ncclResult_t ncclGetPeerDevicePointer(
    ncclWindow_t window,
    size_t offset,
    int peer,
    void** outPtr
);
```

**Key Design Elements:**
- **Automatic Communicator Discovery**: Functions use global communicator registry to find the communicator associated with the window
- **Window Map Optimization**: Global address map (`ncclWindowMap`) for O(1) window-to-communicator lookup
- **Thread-Safe Design**: Protected by `ncclWindowMapMutex` with RAII locking patterns
- Input validation for null pointers and invalid arguments
- Use of existing `ncclShadowPoolToHost` mechanism for window conversion
- Leveraging existing `ncclDevrGetLsaTeamPtrMC` and `ncclDevrGetLsaRankPtr` functions
- **Improved rank conversion**: Uses `ncclTeamRankToTeam` for proper world rank to LSA rank conversion

**Function Distinction:**
- **`ncclGetLsaMultimemDevicePointer`**: Returns multimem base pointer using window's internal multimem
- **`ncclGetMultimemDevicePointer`**: Returns multimem base pointer using provided multimem handle (enables using different multimem handles)
- **`ncclGetLsaDevicePointer`**: Returns pointer to specific LSA peer using LSA rank directly
- **`ncclGetPeerDevicePointer`**: Returns pointer to specific peer using world rank (converts internally to LSA rank using proper team conversion)

### Error Handling

All four functions return `ncclResult_t` error codes to provide comprehensive error reporting:

**Common Error Codes:**
- **`ncclSuccess`**: Operation completed successfully and the pointer is returned in `outPtr`
- **`ncclInvalidArgument`**: Invalid parameters including:
  - NULL window or outPtr
  - Window not found in the window table (invalid window handle)
  - Invalid rank parameters (out of bounds for communicator/LSA team size)
  - Invalid offset (out of window bounds, validated by `ncclGetLsaDevicePointer`)
- **`ncclInternalError`**: Internal errors such as address map lookup failures

**Function-Specific Error Behavior:**

*ncclGetLsaMultimemDevicePointer:*
- Returns `ncclSuccess` with `*outPtr = nullptr` if the system does not support multimem
- This allows graceful degradation without invalidating the communicator
- Users should check if the returned pointer is `nullptr` to determine multimem availability

*ncclGetMultimemDevicePointer:*
- Returns `ncclInvalidArgument` if `multimem.mcBasePtr` is nullptr (invalid multimem handle)
- Returns `ncclSuccess` with `*outPtr = nullptr` if the system does not support multimem
- Allows using external multimem handles instead of relying on window's internal multimem
- Useful when working with multiple multimem contexts or custom multimem allocation

*ncclGetLsaDevicePointer:*
- Validates that `lsaRank` is within bounds: `0 <= lsaRank < lsaSize`
- Validates that `offset` is within window bounds: `0 <= offset < window->size`
- Returns `ncclInvalidArgument` for any out-of-bounds parameter

*ncclGetPeerDevicePointer:*
- Validates that `peer` is within bounds: `0 <= peer < nRanks`
- **Special behavior**: If peer is not reachable via LSA (not in LSA team), returns `ncclSuccess` with `*outPtr = NULL`
- This matches the behavior of the device-side `ncclGetPeerPointer` function

### Window Table Thread Safety

**Problem**: An implementation that relies on querying all communicators is vulnerable to a race condition, since the shadowTable can be modified by a different thread, while the communicators are queried. Locking the communicator list in the RAS subsystem, does not guarantee unchanged shadowTables inside each communicator.

**Solution**: Implemented a global intrusive address map (`ncclWindowMap`) that maps window pointers directly to their associated communicator and host window:

```cpp
std::mutex ncclWindowMapMutex;
ncclIntruAddressMap<ncclWindowData, void*, &ncclWindowData::key, &ncclWindowData::next> ncclWindowMap;
```

The implementation uses NCCL's intrusive address map implementation which avoids per-entry allocations by storing the map metadata (key and next pointer) directly in the `ncclWindowData` structure.

**Benefits**:
- **Thread Safety**: Protected by global mutex with RAII locking patterns
- **Memory Efficiency**: Intrusive design avoids per-entry allocations (only allocates bucket table, not individual entries)

**Implementation Details**:
- Map is populated during window registration (`ncclDevrWindowRegisterInGroup`)
- Map entries are removed during window destruction (`symWindowDestroy`)
- Uses NCCL's intrusive address map implementation (`ncclIntruAddressMap`) for efficient memory management (no per-entry allocations)
- Uses `std::lock_guard` for proper lock scoping with RAII patterns
- Maintains consistency with existing NCCL error handling patterns

<!-- ### System KPIs & Metrics -->
<!-- ### Data Architecture -->
<!-- ### Security Design -->
<!-- ### Debugging & Troubleshooting -->
<!-- ### Logging and Instrumentation -->
<!-- ### Operational Considerations -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->
<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

**Primary Objectives:**
- Validate host-accessible LSA pointer functionality
- Ensure compatibility with existing LSA implementation
- Test both multimem and basic LSA scenarios

### Validation

#### Where to run?
- Multi-GPU systems with LSA support
- Systems with multicast capability for multimem testing
- Development and testing environments

**Current Testing Limitations:**
- Single GPU per MPI rank (no multi-threading testing)
- No testing across different LSA teams to validate peer vs LSA rank distinction

#### What to run?
- `lsa_multimem_pointer_test.cu` - Tests multimem functionality with multicast support using new function names, including `ncclGetMultimemDevicePointer` with multimem handle
- `lsa_pointer_test.cu` - Tests comprehensive LSA and Peer pointer functionality without multicast requirements using new function names
- Both tests require MPI execution with multiple ranks
- Tests now use the simplified API without requiring communicator parameters
- Tests include validation of the new handle-based multimem function


#### Expected output?
- Successful pointer retrieval for valid inputs (`ncclSuccess` returned)
- Proper error codes for invalid arguments:
  - `ncclInvalidArgument` for NULL pointers, invalid ranks, out-of-bounds offsets, invalid multimem handles, or window not found
  - `ncclSuccess` with `*outPtr = nullptr` for multimem operations on systems without multimem support
  - `ncclSuccess` with NULL pointer for peers not reachable via LSA (in `ncclGetPeerDevicePointer`)
- Consistent behavior between device and host functions
- Validation that retrieved pointers match expected LSA memory regions
- Cross-validation: Host-retrieved pointers match device-retrieved pointers
- Cross-validation: `ncclGetLsaMultimemDevicePointer` and `ncclGetMultimemDevicePointer` return identical pointers when using same underlying multimem

**Test Requirements:**
- Must be run with MPI (`mpirun -np <num_ranks>`)
- Requires at least 2 ranks for LSA functionality
- Multimem test requires multicast support and 2+ ranks
- Each rank uses GPU ID = local_rank
- Requires exactly 1 GPU per rank
- Tests validate both LSA and Peer pointer functionality
- Cross-validation between host and device pointer functions

### Performance

**Performance Impact:**
- No direct performance impact on NCCL operations
- Functions are designed for occasional use, not frequent retrieval
- Pointer lifetime matches the shorter of Window and Communicator lifetime

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Summary</h2></summary>
<!-- ============================================================================================-->

This branch successfully extends NCCL's LSA functionality to provide host-accessible pointer retrieval through three distinct functions, enabling seamless integration with existing custom communication kernels. The implementation provides clear separation between LSA rank and world rank operations, maintains full backward compatibility, and includes comprehensive test coverage with cross-validation between host and device functions.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Ludwig Schneider <lschneider@nvidia.com>

</details>
