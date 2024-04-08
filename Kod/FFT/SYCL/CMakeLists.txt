cmake_minimum_required(VERSION 3.0)
# Set the Intel oneAPI compiler
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -I/opt/intel/oneapi/compiler/2024.1/include/sycl -std=c++20")
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)
# Add the include directory to all targets
include_directories(/opt/intel/oneapi/compiler/2024.1/include)
project(SYCL_Project LANGUAGES CXX)
# Add your source files here
set(SOURCES 
    src/sycl_learning.cpp
)

# Create the executable target
add_executable(sycl_learning ${SOURCES})
