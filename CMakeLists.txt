cmake_minimum_required(VERSION 3.25) # ipp6 is using 3.28

# Version information
# Read makefiles/version.mk file
file(READ ${CMAKE_SOURCE_DIR}/makefiles/version.mk VERSION_CONTENT)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]+).*" "\\1" NCCL_SUFFIX "${VERSION_CONTENT}")
string(REGEX REPLACE ".*PKG_REVISION[ ]*:=[ ]*([0-9]+).*" "\\1" PKG_REVISION "${VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

add_compile_definitions(NCCL_USE_CMAKE)
set(ENV{NCCL_USE_CMAKE} "1")

project(NCCL VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
        LANGUAGES CUDA CXX C)

option(VERBOSE "Enable verbose output" OFF)
option(KEEP "Keep intermediate files" OFF)
option(DEBUG "Enable debug build" OFF)
option(ASAN "Enable Address Sanitizer" OFF)
option(UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(TRACE "Enable tracing" OFF)
option(WERROR "Treat warnings as errors" OFF)
option(PROFAPI "Enable profiling API" ON)
option(NVTX "Enable NVTX" ON)
option(RDMA_CORE "Enable RDMA core" OFF)
option(NET_PROFILER "Enable network profiler" OFF)
option(MLX5DV "Enable MLX5DV" OFF)
option(MAX_EXT_NET_PLUGINS "Maximum external network plugins" 0)

find_package(CUDAToolkit REQUIRED)

# CUDA version detection
string(REGEX MATCH "([0-9]+\\.[0-9]+)" CUDA_VERSION "${CUDAToolkit_VERSION}")

# CUDA architecture flags
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting default values based on CUDA version")

    if(${CUDAToolkit_CUDA_MAJOR} EQUAL 9)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
    elseif(${CUDAToolkit_CUDA_MAJOR} EQUAL 10)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
    elseif(${CUDAToolkit_CUDA_MAJOR} EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80")
    elseif(${CUDAToolkit_CUDA_MAJOR} EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80;90")
    elseif(${CUDAToolkit_CUDA_MAJOR} EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80;90;100;120")
    elseif(${CUDAToolkit_CUDA_MAJOR} GREATER 13)
        # For future CUDA versions, include all architectures up to the latest known
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80;90;100;120")
    else()
        # Fallback for older CUDA versions (less than 9)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61")
    endif()
endif()
message(STATUS "Using CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all -fPIC")

# Sanitizer options
if(ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -static-libasan")
endif()

if(UBSAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined -static-libubsan")
endif()

# Additional options
if(TRACE)
    add_definitions(-DENABLE_TRACE)
endif()

if(NOT NVTX)
    add_definitions(-DNVTX_DISABLE)
endif()

if(WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if(PROFAPI)
    add_definitions(-DPROFAPI)
endif()

set(EXTRA_LIBS)

# RDMA and MLX5DV are Linux-specific features
if(RDMA_CORE)
    add_definitions(-DNCCL_BUILD_RDMA_CORE=1)
    find_library(VERBS_LIBRARY NAMES verbs)
    if(VERBS_LIBRARY)
        list(APPEND EXTRA_LIBS ${VERBS_LIBRARY})
    endif()
endif()

if(MLX5DV)
    add_definitions(-DNCCL_BUILD_MLX5DV=1)
    find_library(MLX5_LIBRARY NAMES mlx5)
    if(MLX5_LIBRARY)
        list(APPEND EXTRA_LIBS ${MLX5_LIBRARY})
    endif()
endif()

if(NET_PROFILER)
    add_definitions(-DNCCL_ENABLE_NET_PROFILING=1)
endif()

if(MAX_EXT_NET_PLUGINS GREATER 0)
    add_definitions(-DNCCL_NET_MAX_PLUGINS=${MAX_EXT_NET_PLUGINS})
endif()

# Library dependencies
find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
    list(APPEND EXTRA_LIBS ${RT_LIBRARY})
endif()

# Debug/Release specific flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -G -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

add_subdirectory(ext-profiler/example)
add_subdirectory(src)
add_subdirectory(test)
