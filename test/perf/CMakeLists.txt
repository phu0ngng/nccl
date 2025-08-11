# Performance tests CMake configuration
set(PERF_TEST_SOURCES
    ft_test.cu
    util.cu
    common.cu
    timer.cc
)

set(COLLS
    all_reduce
    all_gather
    broadcast
    reduce_scatter
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
        ${CMAKE_SOURCE_DIR}/ext-profiler/example
        ${CMAKE_SOURCE_DIR}/ext-profiler/example/nccl
    )
    if(MPI_FOUND)
        add_compile_definitions(MPI_SUPPORT)
        target_include_directories(${COLL}_perf PRIVATE ${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(${COLL}_perf PRIVATE nccl ${CUDAToolkit_LIBRARIES} verifiable nccl-profiler-example ${MPI_CXX_LIBRARIES})
    else()
        target_link_libraries(${COLL}_perf PRIVATE nccl ${CUDAToolkit_LIBRARIES} verifiable nccl-profiler-example)
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
