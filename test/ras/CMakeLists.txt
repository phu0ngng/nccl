# RAS tests CMake configuration
if(NOT MPI_FOUND)
    message(WARNING "MPI not found, skipping RAS tests")
    return()
endif()

set(RAS_TEST_SOURCES
    triple_comm_idle.cu
    triple_comm_latecoll.cu
    triple_comm_longidle.cu
    single_comm_mixedcoll.cu
    single_comm_idle.cu
    single_comm_1exit.cu
    single_comm_1suspend.cu
    single_comm_2suspend.cu
    single_comm_4suspend.cu
)

# Add each test as a separate executable
foreach(TEST_SOURCE ${RAS_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/ras
    )

    # Set include directories
    target_include_directories(${TEST_NAME}
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )

    # Link with NCCL, MPI and other required libraries
    target_link_libraries(${TEST_NAME}
        PRIVATE
        nccl
        ${CUDAToolkit_LIBRARIES}
        MPI::MPI_CXX
    )
endforeach()
