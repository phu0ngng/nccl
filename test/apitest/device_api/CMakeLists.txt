# Device API Tests CMake configuration

set(DEVICE_API_SOURCES
    ginSignal_test.cu
)

# Common source file needed by all tests
set(COMMON_SOURCE
    ${CMAKE_SOURCE_DIR}/test/apitest/ncclCommon_test.cu
)

# Create a single executable with all device API tests
add_executable(device_api_test ${DEVICE_API_SOURCES} ${COMMON_SOURCE})

target_include_directories(device_api_test
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/test/apitest
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

