# MPI tests CMake configuration
if(NOT MPI_FOUND)
    message(WARNING "MPI not found, skipping MPI tests")
    return()
endif()

set(MPI_TEST_SOURCES
    mpi_perf.cu
)

# Add each test as a separate executable
foreach(TEST_SOURCE ${MPI_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/mpi
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
        curand
        MPI::MPI_CXX
    )
endforeach()
