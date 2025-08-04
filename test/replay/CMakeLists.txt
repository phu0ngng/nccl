# Replay tests CMake configuration
if(MPI_FOUND)
    set(REPLAY_TEST_SOURCES
        replay.cu
    )

    # Create replay executable
    add_executable(replay ${REPLAY_TEST_SOURCES})
    set_target_properties(replay PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/replay
    )

    # Set include directories
    target_include_directories(replay
        PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )

    # Link with NCCL, MPI and other required libraries
    target_link_libraries(replay
        PRIVATE
        nccl
        ${CUDAToolkit_LIBRARIES}
        MPI::MPI_CXX
        verifiable
    )

    # Copy traces directory to build directory
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/test/replay/traces
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/traces
            ${CMAKE_BINARY_DIR}/test/replay/traces
        DEPENDS ${REPLAY_TEST_SOURCES}
    )

    add_custom_target(copy_traces ALL
        DEPENDS ${CMAKE_BINARY_DIR}/test/replay/traces
    )
else()
    message(WARNING "MPI not found, skipping replay tests")
endif()
