# Verifiable tests CMake configuration
set(VERIFIABLE_LIB_SOURCES
    verifiable.cu
)

set(VERIFIABLE_TEST_SOURCES
    self_test.cu
)

# Add cooperative groups support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# Create verifiable library
add_library(verifiable STATIC ${VERIFIABLE_LIB_SOURCES})
target_include_directories(verifiable PUBLIC
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(verifiable PRIVATE nccl ${CUDAToolkit_LIBRARIES})

# Add each test as a separate executable
foreach(TEST_SOURCE ${VERIFIABLE_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/verifiable
    )

    # Link with NCCL and other required libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        nccl
        ${CUDAToolkit_LIBRARIES}
        verifiable
    )
endforeach()
