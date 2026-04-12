# API tests CMake configuration

# Check CMAKE_CUDA_FLAGS for fp8 multimem architectures that match multimem__funcs.h logic:
# __CUDA_ARCH_SPECIFIC__ == 1000 (sm_100a), 1010 (sm_101a), 1200 (sm_120a), 1210 (sm_121a)
# __CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 (sm_100f+), 1010 (sm_101f+)
# Check for both compute_XXX AND sm_XXX in CMAKE_CUDA_FLAGS to match gencode flags
# If both are present, set compile-time macros to indicate fp8 multimem is available
# Priority: sm_100f > sm_100a > sm_101f > sm_101a > sm_120a > sm_121a
if(CMAKE_CUDA_FLAGS)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_100f" FOUND_COMPUTE_100F)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_100f" FOUND_SM_100F)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_100a" FOUND_COMPUTE_100A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_100a" FOUND_SM_100A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_101f" FOUND_COMPUTE_101F)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_101f" FOUND_SM_101F)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_101a" FOUND_COMPUTE_101A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_101a" FOUND_SM_101A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_120a" FOUND_COMPUTE_120A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_120a" FOUND_SM_120A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "compute_121a" FOUND_COMPUTE_121A)
    string(FIND "${CMAKE_CUDA_FLAGS}" "sm_121a" FOUND_SM_121A)

    # Check for sm_100f (both compute_100f AND sm_100f must be present)
    if(NOT FOUND_COMPUTE_100F EQUAL -1 AND NOT FOUND_SM_100F EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM100F=1)
    # Check for sm_100a (both compute_100a AND sm_100a must be present)
    elseif(NOT FOUND_COMPUTE_100A EQUAL -1 AND NOT FOUND_SM_100A EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM100A=1)
    # Check for sm_101f (both compute_101f AND sm_101f must be present)
    elseif(NOT FOUND_COMPUTE_101F EQUAL -1 AND NOT FOUND_SM_101F EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM101F=1)
    # Check for sm_101a (both compute_101a AND sm_101a must be present)
    elseif(NOT FOUND_COMPUTE_101A EQUAL -1 AND NOT FOUND_SM_101A EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM101A=1)
    # Check for sm_120a (both compute_120a AND sm_120a must be present)
    elseif(NOT FOUND_COMPUTE_120A EQUAL -1 AND NOT FOUND_SM_120A EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM120A=1)
    # Check for sm_121a (both compute_121a AND sm_121a must be present)
    elseif(NOT FOUND_COMPUTE_121A EQUAL -1 AND NOT FOUND_SM_121A EQUAL -1)
        add_compile_definitions(NCCL_BUILD_SM121A=1)
    endif()
endif()

# Permit unsafe device code in apitest (explicit opt-in).
add_compile_definitions(NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1)
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
    ncclCommRevoke_test.cu
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
    ncclCommQueryProperties_test.cu
    ncclCommGrow_test.cu
    ncclDevCommCreate_test.cu
    ncclCommWindowRegister_test.cu
    ncclOneSidedRma_test.cu
)

# Plugin source files
set(PLUGIN_SOURCES
    plugin/net-plugin_v6.cc
    plugin/net-plugin_v7.cc
    plugin/net-plugin_v8.cc
    plugin/net-plugin_v9.cc
    plugin/net-plugin_v10.cc
    plugin/net-plugin_v11.cc
    plugin/net-plugin_v12.cc
    plugin/net-plugin_nodev_v6.cc
    plugin/net-plugin_nodev_v7.cc
    plugin/net-plugin_nodev_v8.cc
    plugin/net-plugin_nodev_v9.cc
    plugin/net-plugin_nodev_v10.cc
    plugin/net-plugin_nodev_v11.cc
    plugin/net-plugin_nodev_v12.cc
    plugin/shared-plugins.cc
    plugin/net-plugin-init-once.cc
    plugin/gin-plugin_v11.cc
)

# Environment plugin source (built from plugins/env/example)
set(ENV_PLUGIN_SOURCE ${CMAKE_SOURCE_DIR}/plugins/env/example/plugin.c)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    add_dependencies(${obj_name} nccl_header)

    # Set CUDA properties for the object library
    set_target_properties(${obj_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
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
        ${CMAKE_SOURCE_DIR}/plugins/net/example/nccl
        ${CMAKE_SOURCE_DIR}/plugins/profiler/example/nccl
        ${CMAKE_SOURCE_DIR}/plugins/tuner/example/nccl
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

# Build environment plugin
add_library(nccl-env-plugin SHARED ${ENV_PLUGIN_SOURCE})

# Set output directory and properties for environment plugin
set_target_properties(nccl-env-plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest/plugin
    PREFIX "libnccl-"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Set include directories for environment plugin
target_include_directories(nccl-env-plugin PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CMAKE_SOURCE_DIR}/plugins/env/example/nccl
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl-env-plugin nccl_header)

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

add_subdirectory(device_api)
