#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
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

  size_t num_bits = std::log2(size);
  bit_reverse_indices(size, num_bits, input_array);

  for (unsigned long step = 1; step <= num_bits; ++step) {
    unsigned long step_size = 1 << step;
    Complex omega =
        std::pow(std::exp(1), (2 * M_PI * J) / static_cast<double>(step_size));

    for (unsigned long start = 0; start < size; start += step_size) {
      for (unsigned long i = 0; i < step_size / 2; i++) {
        unsigned long index_even = start + i;
        unsigned long index_odd = start + i + step_size / 2;
        Complex temp = input_array[index_even];
        input_array[index_even] = input_array[index_even] +
                                  std::pow(omega, i) * input_array[index_odd];
        input_array[index_odd] =
            temp - std::pow(omega, i) * input_array[index_odd];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  unsigned long input_size = std::pow(2, 16);
  Complex *input_array;

  if (argc == 2) {
    std::string path = argv[1];
    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile.is_open()) {
      std::cerr << "Error: Failed to open file for reading.\n";
      return -1;
    }

    inputFile.seekg(0, std::ios::end);
    std::streampos fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    input_size = fileSize / sizeof(Complex);

    input_array = new Complex[input_size];
    inputFile.read(reinterpret_cast<char *>(input_array), fileSize);

    inputFile.close();
  } else {
    input_array = generate_input_array(input_size);
  }

  fft(input_array, input_size);
  std::ofstream outputFile("output.dat", std::ios::binary);
  if (!outputFile.is_open()) {
    std::cerr << "Error: Failed to open file for writing.\n";
    return -1;
  }

  outputFile.write(reinterpret_cast<const char *>(input_array),
                   input_size * sizeof(Complex));

  outputFile.close();

  //  for (long i = 0; i < input_size; i++) {
  //    std::cout << input_array[i] << '\t';
  //  }
  //  std::cout << '\n';

  //  for (long i = 0; i < input_size; i++) {
  //    std::cout << std::abs(output_array[bit_reversal_table[i]]) << '\t';
  //  }
  //  std::cout << '\n';

  delete[] input_array;

  return 0;
}
