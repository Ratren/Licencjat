cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CC_COMPILER "icx")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(Monte_Carlo_distributed-ranges LANGUAGES CXX)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fsycl COMPILER_SUPPORTS_FSYCL)

if(NOT COMPILER_SUPPORTS_FSYCL)
  message(
    FATAL_ERROR
      "A compiler with SYCL support is required."
  )
endif()

find_package(MPI REQUIRED)

add_subdirectory(src)

option(ENABLE_CUDA "Build for cuda" OFF)
option(ENABLE_FORMAT "Build with format library" ON)

include(FetchContent)

FetchContent_Declare(
  distributed-ranges
  GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
  GIT_TAG main 
)
FetchContent_MakeAvailable(distributed-ranges)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(cpp-format)

