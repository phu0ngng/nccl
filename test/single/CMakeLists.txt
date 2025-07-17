# Single tests CMake configuration
set(SINGLE_TEST_SOURCES
    reduce_test.cu
    all_reduce_test.cu
    broadcast_test.cu
    reduce_scatter_test.cu
    stress_test.cu
    abort_gpu_per_process.cu
    abort_single_process.cu
    all_gather_test.cu
)

# Add each test as a separate executable
foreach(TEST_SOURCE ${SINGLE_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/single
    )

    # Set include directories
    target_include_directories(${TEST_NAME}
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/test/include
    )

    # Link with NCCL and other required libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        nccl
        ${CUDAToolkit_LIBRARIES}
        curand
        ${CUDAToolkit_LIBRARIES}
    )
endforeach()
