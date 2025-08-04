# Stress tests CMake configuration
set(STRESS_TEST_SOURCES
    stress.cu
)

# Create stress test executable
add_executable(stress ${STRESS_TEST_SOURCES})
set_target_properties(stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/stress
)

# Set include directories
target_include_directories(stress
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
)

# Link with NCCL and other required libraries
target_link_libraries(stress
    PRIVATE
    nccl
    ${CUDAToolkit_LIBRARIES}
)

# Copy support files to build directory using POST_BUILD command
add_custom_command(
    TARGET stress
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_random_alltoallv.sh
        ${CMAKE_BINARY_DIR}/test/stress/gen_random_alltoallv.sh
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/test.txt
        ${CMAKE_BINARY_DIR}/test/stress/test.txt
    COMMENT "Copying support files for stress test"
)
