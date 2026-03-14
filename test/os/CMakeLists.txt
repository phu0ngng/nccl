# OS abstraction layer for NCCL tests

# Create OS library with platform-specific implementations
if(NCCL_OS_WINDOWS)
    set(OS_SOURCES windows.cc)
elseif(NCCL_OS_LINUX)
    set(OS_SOURCES linux.cc)
endif()

add_library(nccl_test_os STATIC ${OS_SOURCES})

# Set include directories
target_include_directories(nccl_test_os
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
