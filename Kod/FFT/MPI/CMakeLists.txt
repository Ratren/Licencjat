cmake_minimum_required(VERSION 3.0)
# Set the Intel oneAPI compiler
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Add the include directory to all targets
project(MPI_fft LANGUAGES CXX)
# Add your source files here
find_package(MPI REQUIRED)
set(SOURCES 
    src/MPI_fft.cpp
)

# Create the executable target
add_executable(MPI_fft ${SOURCES})
target_link_libraries(MPI_fft MPI::MPI_CXX)
