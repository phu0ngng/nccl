cmake_minimum_required(VERSION 3.10...3.27)
project(port-failure-emu VERSION 0.2.0 LANGUAGES C)

include(CTest)
enable_testing()

# Find RDMA libraries (optional - skip build if not available)
find_library(MLX5_LIBRARY NAMES mlx5)
find_library(VERBS_LIBRARY NAMES ibverbs)

if(NOT MLX5_LIBRARY OR NOT VERBS_LIBRARY)
    message(STATUS "RDMA libraries not found (libmlx5: ${MLX5_LIBRARY}, libibverbs: ${VERBS_LIBRARY})")
    message(STATUS "Skipping port-failure-emu build (requires RDMA hardware support)")
    return()
endif()

message(STATUS "Building port-failure-emu with RDMA support")

# Find headers (typically in /usr/include/infiniband)
find_path(VERBS_INCLUDE_DIR infiniband/verbs.h)
find_path(MLX5DV_INCLUDE_DIR infiniband/mlx5dv.h)

if(NOT VERBS_INCLUDE_DIR OR NOT MLX5DV_INCLUDE_DIR)
    message(STATUS "RDMA development headers not found (infiniband/verbs.h: ${VERBS_INCLUDE_DIR}, infiniband/mlx5dv.h: ${MLX5DV_INCLUDE_DIR})")
    message(STATUS "Skipping port-failure-emu build (requires RDMA development packages)")
    return()
endif()

include_directories("include")
include_directories(${VERBS_INCLUDE_DIR})
include_directories(${MLX5DV_INCLUDE_DIR})

# ============================================================================
#                         Shared Library Build
# ============================================================================

# Library source files (excluding CLI and examples)
set(LIB_SOURCES
    src/pfe_core.c
    src/pfe_flow_table.c
    src/pfe_device.c
)

# Create shared library
add_library(portfailureemu SHARED ${LIB_SOURCES})
target_link_libraries(portfailureemu ${MLX5_LIBRARY} ${VERBS_LIBRARY})
set_target_properties(portfailureemu PROPERTIES
    OUTPUT_NAME "portfailureemu"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER "include/port_failure_emu.h"
)
target_compile_options(portfailureemu PRIVATE -Wall -Wextra -ggdb)

# ============================================================================
#                         CLI Tool Build
# ============================================================================

# Create CLI executable
add_executable(portFailureEmu cli/main.c)
set_target_properties(portFailureEmu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_options(portFailureEmu PRIVATE -Wall -Wextra -ggdb)
target_link_libraries(portFailureEmu portfailureemu)

# ============================================================================
#                         Examples Build
# ============================================================================

# Simple example
add_executable(portFailureEmuExample examples/port_failure_emu_example.c)
set_target_properties(portFailureEmuExample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_options(portFailureEmuExample PRIVATE -Wall -Wextra -ggdb)
target_link_libraries(portFailureEmuExample portfailureemu)

# ============================================================================
#                         Installation Rules
# ============================================================================

# Install library
install(TARGETS portfailureemu
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Install CLI tool
install(TARGETS portFailureEmu
    RUNTIME DESTINATION bin
)

# Install examples (optional)
install(TARGETS portFailureEmuExample
    RUNTIME DESTINATION share/port-failure-emu/examples
)

# ============================================================================
#                         CPack Configuration
# ============================================================================

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
