# Device API Tests CMake configuration

set(DEVICE_API_SOURCES
    ginSignal_test.cu
    ginVaSignal_test.cu
    ginPut_test.cu
    ginPutValue_test.cu
    reduceCopy/reference_gpu.cu
    reduceCopy/test_matrix_listener.cu
)

# Common source file needed by all tests
set(COMMON_SOURCE
    ${CMAKE_SOURCE_DIR}/test/apitest/ncclCommon_test.cu
)

# reduceCopy test generation (device API tests)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(NCCL_TEST_REDUCE_COPY_TYPES "$ENV{NCCL_TEST_REDUCE_COPY_TYPES}" CACHE STRING "Comma-separated reduceCopy types to generate (default empty)")
set(REDUCECOPY_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/reduceCopy/generated)
file(MAKE_DIRECTORY ${REDUCECOPY_GENERATED_DIR})
set(REDUCECOPY_GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/reduceCopy/generate_tests.py)
set(REDUCECOPY_GENERATED_STAMP ${REDUCECOPY_GENERATED_DIR}/generated.stamp)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_TEST_REDUCE_COPY_TYPES=${NCCL_TEST_REDUCE_COPY_TYPES}
            ${Python3_EXECUTABLE} ${REDUCECOPY_GEN_SCRIPT}
            --output-dir ${REDUCECOPY_GENERATED_DIR}
            --list-files
    OUTPUT_VARIABLE REDUCECOPY_GENERATED_FILES_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE " " ";" REDUCECOPY_GENERATED_SOURCES "${REDUCECOPY_GENERATED_FILES_RAW}")

# List generated sources as OUTPUT (not BYPRODUCTS) so CMake creates proper
# dependencies: compile rules will wait for the generator to run first.
add_custom_command(
    OUTPUT ${REDUCECOPY_GENERATED_STAMP} ${REDUCECOPY_GENERATED_SOURCES}
    COMMAND ${CMAKE_COMMAND} -E env NCCL_TEST_REDUCE_COPY_TYPES=${NCCL_TEST_REDUCE_COPY_TYPES}
            ${Python3_EXECUTABLE} ${REDUCECOPY_GEN_SCRIPT}
            --output-dir ${REDUCECOPY_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${REDUCECOPY_GENERATED_STAMP}
    DEPENDS
        ${REDUCECOPY_GEN_SCRIPT}
        ${CMAKE_CURRENT_SOURCE_DIR}/reduceCopy/test_reduce_copy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/reduceCopy/config.h
    COMMENT "Generating reduceCopy test sources"
)
add_custom_target(reducecopy_generate_tests_device DEPENDS ${REDUCECOPY_GENERATED_STAMP} ${REDUCECOPY_GENERATED_SOURCES})

list(APPEND DEVICE_API_SOURCES ${REDUCECOPY_GENERATED_SOURCES})
set_source_files_properties(${REDUCECOPY_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# Create a single executable with all device API tests (including reduceCopy)
add_executable(device_api_test ${DEVICE_API_SOURCES} ${COMMON_SOURCE})

add_dependencies(device_api_test reducecopy_generate_tests_device)

target_include_directories(device_api_test
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/test/apitest
    ${CMAKE_SOURCE_DIR}/test/googletest/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/reduceCopy
)

set_target_properties(device_api_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/apitest/device_api
)

target_link_libraries(device_api_test
    PRIVATE
    nccl
    ${CUDAToolkit_LIBRARIES}
    gtest
    gtest_main
)

