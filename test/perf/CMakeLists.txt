# Performance tests CMake configuration

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

# Permit unsafe device code in perf tests (explicit opt-in).
add_compile_definitions(NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1)

set(PERF_TEST_SOURCES
    ft_test.cu
    util.cu
    common.cu
    timer.cc
)

set(COLLS
    all_reduce
    all_gather
    all_gatherv
    broadcast
    reduce_scatter
    reduce_scatterv
    reduce
    alltoall
    gather
    scatter
    sendrecv
    hypercube
)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

foreach(COLL IN LISTS COLLS)
    add_executable(${COLL}_perf ${COLL}.cu ${PERF_TEST_SOURCES})
    set_target_properties(${COLL}_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/perf
    )
    # Set include directories
    target_include_directories(${COLL}_perf
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/plugins/profiler/example
        ${CMAKE_SOURCE_DIR}/plugins/profiler/example/nccl
    )
    if(MPI_FOUND)
        add_compile_definitions(MPI_SUPPORT)
        target_include_directories(${COLL}_perf PRIVATE ${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(${COLL}_perf PRIVATE nccl ${CUDAToolkit_LIBRARIES} verifiable ${MPI_CXX_LIBRARIES})
    else()
        target_link_libraries(${COLL}_perf PRIVATE nccl ${CUDAToolkit_LIBRARIES} verifiable)
    endif()
    add_dependencies(${COLL}_perf verifiable nccl-profiler-example)
endforeach()

# Copy replay-coll.log to build directory
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/test/perf/replay-coll.log
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/replay-coll.log
        ${CMAKE_BINARY_DIR}/test/perf/replay-coll.log
    DEPENDS ${PERF_TEST_SOURCES}
)

add_custom_target(copy_replay_log ALL
    DEPENDS ${CMAKE_BINARY_DIR}/test/perf/replay-coll.log
)
