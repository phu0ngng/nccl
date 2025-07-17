find_package(MPI QUIET)

# Add cooperative groups support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Google Test setup
set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GTEST_DIR}/include
    ${GTEST_DIR}
)

# Build Google Test
add_library(gtest STATIC
    ${GTEST_DIR}/src/gtest-all.cc
)
add_library(gtest_main STATIC
    ${GTEST_DIR}/src/gtest_main.cc
)
target_link_libraries(gtest_main gtest)

# Add test subdirectories
add_subdirectory(apitest)
add_subdirectory(single)
add_subdirectory(unit)
add_subdirectory(perf)
add_subdirectory(examples)
add_subdirectory(binary)
add_subdirectory(stress)
add_subdirectory(verifiable)

# Add MPI-dependent directories if MPI is found
if(MPI_FOUND)
    add_subdirectory(mpi)
    add_subdirectory(replay)
    add_subdirectory(ras)
endif()
