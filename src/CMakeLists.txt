# Source files
set(LIBSRCFILES
    bootstrap.cc
    channel.cc
    collectives.cc
    debug.cc
    enqueue.cc
    group.cc
    init.cc
    init_nvtx.cc
    proxy.cc
    transport.cc
    mnnvl.cc
    allocator.cc
    symmetric.cc
)

# Add compatibility shim if using static cudart
if(CUDARTLIB STREQUAL "cudart_static")
    list(APPEND LIBSRCFILES enhcompat.cc)
endif()

# Configure header file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in
    ${CMAKE_BINARY_DIR}/include/nccl.h
    @ONLY
)

# Configure pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl.pc.in
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc
    @ONLY
)

# Add files from subdirectories
add_subdirectory(transport)
add_subdirectory(misc)
add_subdirectory(register)
add_subdirectory(graph)
add_subdirectory(plugin)
add_subdirectory(device)
add_subdirectory(ras)

# Add all source files
list(APPEND LIBSRCFILES
    ${TRANSPORT_SOURCES}
    ${MISC_SOURCES}
    ${REGISTER_SOURCES}
    ${GRAPH_SOURCES}
    ${PLUGIN_SOURCES}
    ${RAS_SOURCES}
)

# # Set CUDA specific flags
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xptxas -maxrregcount=128")

# Add CUDA architecture flags
# set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for")

###################### Create a shared NCCL library ############################
add_library(nccl SHARED)

target_sources(nccl PRIVATE ${LIBSRCFILES})

# Include directories
target_include_directories(nccl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${CMAKE_CURRENT_BINARY_DIR}/device/gensrc
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set version and output name
set_target_properties(nccl PROPERTIES
    VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
    SOVERSION ${NCCL_MAJOR}
    OUTPUT_NAME "nccl"
    PREFIX "lib"
)

# Set CUDA specific flags
set_target_properties(nccl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(nccl
    PRIVATE
    nccl_device
    pthread
    rt
    dl
    cudart
    cuda
    ${CUDAToolkit_LIBRARIES}
    ${EXTRA_LIBS}
)

# Set output directories for nccl shared library
set_target_properties(nccl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

###################### Create a ras binary executable ############################
set(RAS_BINSRCFILES ras/client.cc)

add_executable(ncclras ${RAS_BINSRCFILES})

target_include_directories(ncclras PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ncclras
    PRIVATE
    pthread
    rt
    dl
)

# Set output directory for ncclras executable
set_target_properties(ncclras PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

###################### Create a static NCCL library ############################
add_library(nccl_static STATIC ${LIBSRCFILES})

# Include directories
target_include_directories(nccl_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${CMAKE_CURRENT_BINARY_DIR}/device/gensrc
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(nccl_static
    PRIVATE
    nccl_device
    pthread
    rt
    dl
    cudart
    cuda
    ${CUDAToolkit_LIBRARIES}
    ${EXTRA_LIBS}
)

# Set CUDA specific flags
set_target_properties(nccl_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Set output directory for nccl_static library
set_target_properties(nccl_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
