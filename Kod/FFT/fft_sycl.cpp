#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>
#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

void generate_bit_reversal_table(unsigned long num_bits,
                                 unsigned long *bit_reversal_table) {
  unsigned long tableSize = 1 << num_bits;
  for (unsigned long i = 0; i < tableSize; ++i) {
    unsigned long reversed = 0;
    for (unsigned long j = 0; j < num_bits; ++j) {
      if (i & (1 << j)) {
        reversed |= (1 << (num_bits - 1 - j));
      }
    }
    bit_reversal_table[i] = reversed;
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
Complex *fft(Complex *input_array, unsigned long size) {

  Complex *output_array = new Complex[size];
  std::copy(input_array, input_array + size, output_array);

  long num_bits = static_cast<long>(std::log2(size));
  unsigned long *bit_reversal_table = new unsigned long[1 << num_bits];
  generate_bit_reversal_table(num_bits, bit_reversal_table);

  for (unsigned long step = 1; step <= num_bits; ++step) {
    unsigned long step_size = 1 << step;
    Complex omega =
        std::pow(std::exp(1), (2 * M_PI * J) / static_cast<double>(step_size));

    for (unsigned long start = 0; start < size; start += step_size) {
      for (unsigned long i = 0; i < step_size / 2; i++) {
        unsigned long index_even = bit_reversal_table[start + i];
        unsigned long index_odd = bit_reversal_table[start + i + step_size / 2];
        Complex temp = output_array[index_even];
        output_array[index_even] = output_array[index_even] +
                                   std::pow(omega, i) * output_array[index_odd];
        output_array[index_odd] =
            temp - std::pow(omega, i) * output_array[index_odd];
      }
    }
  }

  delete[] bit_reversal_table;

  return output_array;
}

int main() {
  unsigned long input_size = std::pow(2, 20);

  Complex *input_array = generate_input_array(input_size);

  Complex *output_array = fft(input_array, input_size);

  //  for (long i = 0; i < input_size; i++) {
  //    std::cout << input_array[i] << '\t';
  //  }
  //  std::cout << '\n';

  long num_bits = static_cast<long>(std::log2(input_size));
  unsigned long *bit_reversal_table = new unsigned long[1 << num_bits];
  generate_bit_reversal_table(num_bits, bit_reversal_table);

  //  for (long i = 0; i < input_size; i++) {
  //    std::cout << std::abs(output_array[bit_reversal_table[i]]) << '\t';
  //  }
  //  std::cout << '\n';

  delete[] input_array;
  delete[] output_array;
  delete[] bit_reversal_table;

  return 0;
}
