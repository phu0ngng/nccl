set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-mock SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-mock PRIVATE
    ${CMAKE_BINARY_DIR}/include
)

# Set output name to match Makefile
set_target_properties(nccl-mock PROPERTIES
    OUTPUT_NAME "libnccl-net"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)
