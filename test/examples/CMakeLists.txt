# Examples CMake configuration
set(EXAMPLE_SOURCES
    multiproc_singlegpu_driver.cu
    singleproc_multigpu.cu
    singleproc_multigpu_driver.cu
    cuda_graph.cu
    multiproc_env_id.cu
    multiproc_multigpu.cu
    multiproc_singlegpu.cu
)

# Add MPI include directories if MPI is found
if(MPI_FOUND)
    # Add MPI compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
endif()

# Add each example as a separate executable
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    # Create executable
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/examples
    )

    # Set include directories
    target_include_directories(${EXAMPLE_NAME}
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
    )

    if(MPI_FOUND)
        target_include_directories(${EXAMPLE_NAME}
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
    endif()

    # Link with NCCL and other required libraries
    target_link_libraries(${EXAMPLE_NAME}
        PRIVATE
        nccl
        CUDA::cuda_driver
        ${CUDAToolkit_LIBRARIES}
    )

    set(test_names cuda_graph multiproc_multigpu multiproc_singlegpu multiproc_env_id multiproc_singlegpu_driver singleproc_multigpu singleproc_multigpu_driver)
    # Add MPI libraries if MPI is found and the example needs it
    if(MPI_FOUND AND (${EXAMPLE_NAME} IN_LIST test_names))
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE HAVE_MPI=1)
        # Add MPI link flags
        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
    endif()
endforeach()
