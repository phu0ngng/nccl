# API tests CMake configuration
set(API_TEST_SOURCES
    ncclCommInitRankConfig_test.cu
    ncclCommShrink_test.cu
    ncclDebug_test.cu
    ncclCommInitRank_test.cu
    ncclReduce_test.cu
    ncclSendRecv_test.cu
    ncclReduceScatter_test.cu
    ncclGroup_test.cu
    ncclCommRegister_test.cu
    ncclBroadcast_test.cu
    ncclCommInitAll_test.cu
    ncclAllGather_test.cu
    ncclAllReduce_test.cu
    ncclCommInitRankScalable_test.cu
    ncclCudaGraph_test.cu
    ncclCommon_test.cu
    ncclCommAbort_test.cu
    ncclRedOpCreatePreMulSum.cu
    ncclGetUniqueId_test.cu
    ncclCommSplit_test.cu
    ncclCommFinalize_test.cu
    ncclCommGetAsyncError_test.cu
    ncclCommDestroy_test.cu
    ncclGetErrorString_test.cu
    ncclCommUserRank_test.cu
    ncclCommCount_test.cu
    ncclCommCuDevice_test.cu
    ncclGetVersion_test.cu
    ncclBcast_test.cu
)

# Plugin source files
set(PLUGIN_SOURCES
    plugin/net-plugin_v6.cc
    plugin/net-plugin_v7.cc
    plugin/net-plugin_v8.cc
    plugin/net-plugin_v9.cc
    plugin/net-plugin_v10.cc
    plugin/net-plugin_v11.cc
    plugin/net-plugin_nodev_v6.cc
    plugin/net-plugin_nodev_v7.cc
    plugin/net-plugin_nodev_v8.cc
    plugin/net-plugin_nodev_v9.cc
    plugin/net-plugin_nodev_v10.cc
    plugin/net-plugin_nodev_v11.cc
    plugin/shared-plugins.cc
)

# Create object files for each CUDA source
set(API_TEST_OBJECTS)
foreach(source ${API_TEST_SOURCES})
    get_filename_component(source_name ${source} NAME_WE)
    set(obj_name ${source_name}_obj)

    # Create object library for each CUDA source
    add_library(${obj_name} OBJECT ${source})

    # Set include directories
    target_include_directories(${obj_name} PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../googletest/googletest/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    add_dependencies(${obj_name} nccl_header)

    # Set CUDA properties for the object library
    set_target_properties(${obj_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    # Set CUDA compiler flags - include the common test header
    target_compile_options(${obj_name} PRIVATE
        -include ${CMAKE_CURRENT_SOURCE_DIR}/ncclCommon_test.cuh
    )

    list(APPEND API_TEST_OBJECTS $<TARGET_OBJECTS:${obj_name}>)
endforeach()

# Create the main test executable
add_executable(apitest ${API_TEST_OBJECTS})

# Create the static test executable (equivalent to apitest_static in Makefile)
add_executable(apitest_static ${API_TEST_OBJECTS})

# Set CUDA specific properties for both executables
set_target_properties(apitest apitest_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest
)

# Link with NCCL and other required libraries for dynamic version
target_link_libraries(apitest PRIVATE
    nccl
    ${CUDAToolkit_LIBRARIES}
    gtest
    gtest_main
    pthread
    rt
    dl
)

# Link with NCCL static and other required libraries for static version
target_link_libraries(apitest_static PRIVATE
    nccl_static
    ${CUDAToolkit_LIBRARIES}
    gtest
    gtest_main
    pthread
    rt
    dl
)

# Build plugin libraries
foreach(plugin_source ${PLUGIN_SOURCES})
    get_filename_component(plugin_name ${plugin_source} NAME_WE)

    # Create shared library for each plugin
    add_library(${plugin_name} SHARED ${plugin_source})

    # Set output directory and properties
    set_target_properties(${plugin_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest/plugin
        PREFIX "libnccl-"
        SUFFIX ".so"
        POSITION_INDEPENDENT_CODE ON
    )

    # Set include directories for plugins (prioritize extension paths)
    target_include_directories(${plugin_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/ext-net/example/nccl
        ${CMAKE_SOURCE_DIR}/ext-profiler/example/nccl
        ${CMAKE_SOURCE_DIR}/ext-tuner/example/nccl
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/include/plugin
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )

    add_dependencies(${plugin_name} nccl_header)

    # Set compiler flags
    set_target_properties(${plugin_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endforeach()

# Add test for static version
add_test(NAME apitest_static
    COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/test/apitest/plugin:$ENV{LD_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/test/apitest/apitest_static
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest
)

# Create a custom target to run the test (equivalent to 'make test' in Makefile)
add_custom_target(test_apitest
    COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/test/apitest/plugin:$ENV{LD_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/test/apitest/apitest
    DEPENDS apitest
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest
    COMMENT "Running apitest"
)
