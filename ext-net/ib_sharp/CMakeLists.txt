set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvwrap.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.c
)

# Create shared library
add_library(nccl-ib_sharp SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-ib_sharp PRIVATE
    ${CMAKE_BINARY_DIR}/include
)

# Set output name to match Makefile
set_target_properties(nccl-ib_sharp PROPERTIES
    OUTPUT_NAME "libnccl-net"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)
