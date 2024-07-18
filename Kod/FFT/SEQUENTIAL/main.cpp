#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

void bit_reverse_indices(size_t size, unsigned long num_bits,
                         Complex *input_array) {
  Complex *temp_array = new Complex[size];
  std::copy(input_array, input_array + size, temp_array);
  unsigned long tableSize = 1 << num_bits;
  for (unsigned long i = 0; i < tableSize; ++i) {
    unsigned long reversed = 0;
    for (unsigned long j = 0; j < num_bits; ++j) {
      if (i & (1 << j)) {
        reversed |= (1 << (num_bits - 1 - j));
      }
    }
    input_array[i] = temp_array[reversed];
  }
}

Complex *generate_input_array(unsigned long size) {
  Complex *input_array = new Complex[size];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (unsigned long i = 0; i < size; ++i) {
    double real_part = dis(gen);
    input_array[i] = Complex(real_part, 0);
  }

  return input_array;
}
//
//                    [0, 1, 2, 3, 4, 5, 6, 7]
//                [0, 2, 4, 6]        [1, 3, 5, 7]
//              [0, 4]    [2, 6]    [1, 5]    [3, 7]
//
//
void fft(Complex *input_array, unsigned long size) {

  auto start = std::chrono::high_resolution_clock::now();
  size_t num_bits = std::log2(size);
  bit_reverse_indices(size, num_bits, input_array);

  for (unsigned long i = 1; i <= num_bits; ++i) {
    unsigned long step_size = 1 << i;
    Complex omega = std::exp(-2.0 * J * M_PI / (double) step_size);

    for (unsigned long start = 0; start < size; start += step_size) {
      Complex omega_power = 1;
      for (unsigned long j = 0; j < step_size / 2; j++) {
        unsigned long index_even = start + j;
        unsigned long index_odd = start + j + step_size / 2;
        Complex temp = input_array[index_even];
        input_array[index_even] += omega_power * input_array[index_odd];
        input_array[index_odd] = temp - omega_power * input_array[index_odd];
        omega_power *= omega;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << "FFT zostało wykonane w czasie: " << duration.count() << " sekund'\n";
}

int main(int argc, char *argv[]) {
  unsigned long input_size;
  Complex *input_array;

  std::ifstream inputFile("./TEST_DATA/FFT_test_vector", std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Failed to open file for reading.\n";
    return -1;
  }

  inputFile.seekg(0, std::ios::end);
  long fileSize = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  input_size = fileSize / sizeof(Complex);

  input_array = new Complex[input_size];
  inputFile.read(reinterpret_cast<char *>(input_array), fileSize);

  inputFile.close();

  fft(input_array, input_size);

  /* std::ofstream outputFile("output.dat", std::ios::binary); */
  /* if (!outputFile.is_open()) { */
  /*   std::cerr << "Error: Failed to open file for writing.\n"; */
  /*   return -1; */
  /* } */

  /* outputFile.write(reinterpret_cast<const char *>(input_array), */
  /*                  input_size * sizeof(Complex)); */

  /* outputFile.close(); */

  delete[] input_array;

  return 0;
}
