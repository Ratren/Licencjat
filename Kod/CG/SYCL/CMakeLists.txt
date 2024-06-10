cmake_minimum_required(VERSION 3.0)
# Set the Intel oneAPI compiler
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-fsycl -O3 -std=c++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Add the include directory to all targets
project(SYCL_Project LANGUAGES CXX)
# Add your source files here
set(SOURCES 
    src/CG_SYCL.cpp
)

# Create the executable target
add_executable(CG_SYCL ${SOURCES})
