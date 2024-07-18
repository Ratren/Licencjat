#!/bin/bash

source /opt/intel/oneapi/setvars.sh

rm -rf build
mkdir -p build
mkdir build/TEST_DATA

#BUILD MONTE CARLO
cd MONTE_CARLO/SEQUENTIAL
g++ -O1 -o main main.cpp
cp main ../../build/Monte_Carlo_sequential

cd ../SYCL
cmake -B build
cmake --build build
cp build/main ../../build/Monte_Carlo_SYCL

cd ../MPI
mpicxx -O1 -o main main.cpp
cp main ../../build/Monte_Carlo_MPI

cd ../dist-ranges
cmake -B build
cmake --build build
cp build/src/main ../../build/Monte_Carlo_distributed-ranges

#BUILD CG
cd ../../CG/SEQUENTIAL
g++ -O1 -o main main.cpp
cp main ../../build/CG_sequential

cd ../SYCL
cmake -B build
cmake --build build
cp build/main ../../build/CG_SYCL

cd ../MPI
mpicxx -O1 -o main main.cpp
cp main ../../build/CG_MPI

cd ../dist-ranges
cmake -B build
cmake --build build
cp build/src/main ../../build/CG_distributed-ranges

cd ../GEN_DATA
cmake -B build
cmake --build build
cp build/generate_data ../../build/CG_generate_data

#BUILD FFT
cd ../../FFT/SEQUENTIAL
g++ -O1 -o main main.cpp
cp main ../../build/FFT_sequential

cd ../SYCL
cmake -B build
cmake --build build
cp build/main ../../build/FFT_SYCL

cd ../MPI
mpicxx -O1 -o main main.cpp
cp main ../../build/FFT_MPI

cd ../dist-ranges
cmake -B build
cmake --build build
cp build/src/main ../../build/FFT_distributed-ranges

cd ../GEN_DATA
cmake -B build
cmake --build build
cp build/generate_data ../../build/FFT_generate_data



