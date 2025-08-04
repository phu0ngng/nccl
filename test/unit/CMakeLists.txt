# Unit tests CMake configuration
set(UNIT_TEST_SOURCES
    align_test.cu
    graph_test.cc
    model.cc
    data.cc
    xml_test.cc
    xml_fuse_test.cc
    comm_leak_test.cu
    channel_test.cc
    enqueue_test.cc
    log_algo.cc
    ft_test.cc
    overlap_test.cc
    plugin_load.cc
)

# MPI-dependent sources
if(MPI_FOUND)
    list(APPEND UNIT_TEST_SOURCES
        comm_leak_test_multi.cu
        coll_leak_test_multi.cu
        ft_abort_rank0.cu
    )
endif()

set(SRC_WITH_DYNAMIC_LIBS
    ft_test.cc
    ft_abort_rank0.cu
    device_id_test.cc
)

# Add each test as a separate executable
foreach(TEST_SOURCE ${UNIT_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME}
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/graph/
    )
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit
    )

    if(${TEST_SOURCE} IN_LIST SRC_WITH_DYNAMIC_LIBS)
        target_link_libraries(${TEST_NAME}
            PRIVATE
            nccl
            CUDA::cudart
            gtest
            gtest_main
        )
    else()
        target_link_libraries(${TEST_NAME}
            PRIVATE
            nccl_static
            ${CUDAToolkit_LIBRARIES}
            gtest
            gtest_main
        )
    endif()

    if(MPI_FOUND)
        target_link_libraries(${TEST_NAME}
            PRIVATE
            MPI::MPI_CXX
        )
    endif()
endforeach()

# Add dependency from test executables to copy_topo target
foreach(TEST_SOURCE ${UNIT_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_dependencies(${TEST_NAME} copy_topo)
endforeach()

# Copy topo directory to build directory
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/test/unit/topo
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/topo
        ${CMAKE_BINARY_DIR}/test/unit/topo
    DEPENDS ${UNIT_TEST_SOURCES}
)

add_custom_target(copy_topo ALL
    DEPENDS ${CMAKE_BINARY_DIR}/test/unit/topo
)
