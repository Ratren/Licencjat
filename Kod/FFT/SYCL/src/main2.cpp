#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <oneapi/dpl/random>
#include <queue.hpp>
#include <time.h>
#include <vector>
#include <chrono>
#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

std::vector<Complex> generate_input_array(size_t size) {

  size_t chunkSize = size / 8;
  uint32_t seed = std::time(0);
  std::vector<Complex> array(size);

  sycl::queue queue(sycl::cpu_selector_v);

  sycl::buffer<Complex, 1> buf(array.data(), sycl::range<1>(size));

  for (size_t offset = 0; offset < size; offset += chunkSize) {
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class generate_random>(
          sycl::range<1>(chunkSize), [=](sycl::id<1> idx) {
            size_t globalIdx = idx[0] + offset;
            if (globalIdx < size) {
              oneapi::dpl::minstd_rand engine(seed, globalIdx);
              oneapi::dpl::uniform_real_distribution<double> distr;
              acc[globalIdx] = Complex(distr(engine), 0.0);
            }
          });
    });
  }

  queue.wait_and_throw();

  return array;
}

void bit_reverse_indices(std::vector<Complex> *array_to_sort, size_t size) {
  Complex *sorting_data = new Complex[size];

  size_t chunkSize = size / 8;
  std::copy(array_to_sort->data(), array_to_sort->data() + size, sorting_data);

  sycl::queue queue(sycl::cpu_selector_v);

  sycl::buffer<Complex, 1> buf_from(sorting_data, sycl::range<1>(size));
  sycl::buffer<Complex, 1> buf_to(array_to_sort->data(), sycl::range<1>(size));

  for (size_t offset = 0; offset < size; offset += chunkSize) {
    queue.submit([&](sycl::handler &cgh) {
      auto from = buf_from.get_access<sycl::access::mode::read>(cgh);
      auto to = buf_to.get_access<sycl::access::mode::write>(cgh);
      int num_bits = std::log2(size);
      cgh.parallel_for<class reverse_indices>(
          sycl::range<1>(chunkSize), [=](sycl::id<1> idx) {
            size_t globalIdx = idx[0] + offset;
            if (globalIdx < size) {
              int reversed = 0;
              for (int j = 0; j < num_bits; j++) {
                if (globalIdx & (1 << j)) {
                  reversed |= (1 << (num_bits - 1 - j));
                }
              }
              to[globalIdx] = from[reversed];
            }
          });
    });
  }

  queue.wait();
}

void fft(std::vector<Complex> *input_array, size_t size) {

  sycl::queue queue(sycl::cpu_selector_v);

  sycl::buffer<Complex, 1> buf(input_array->data(), sycl::range<1>(size));

  int num_bits = std::log2(size);

  for (size_t step = 1; step <= num_bits; ++step) {
    const size_t step_size_outer = 1 << step;
    std::cout << input_array->data()[0];
    if ((size / step_size_outer) >= 64) {
      queue.submit([&](sycl::handler &cgh) {
        const size_t step_size = 1 << step;
        const Complex omega = std::exp(-2.0 * J * M_PI / (double)step_size);
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class fft_calculations>(
            sycl::range<1>(size / step_size), [=](sycl::id<1> idx) {
              size_t start = idx[0] * step_size;
              size_t index_even = start;
              size_t index_odd = start + step_size / 2;
              Complex angle = std::pow(omega, 0);
              Complex temp = acc[index_even];
              acc[index_even] = temp + angle * acc[index_odd];
              acc[index_odd] = temp - angle * acc[index_odd];
              for (size_t i = 1; i < step_size / 2; i++) {
                index_even = start + i;
                index_odd = start + i + step_size / 2;
                temp = acc[index_even];
                angle = angle * omega;
                acc[index_even] = temp + angle * acc[index_odd];
                acc[index_odd] = temp - angle * acc[index_odd];
              }
            });
      });
    } else {
      for (size_t start = 0; start < size; start += step_size_outer) {
        queue.submit([&](sycl::handler &cgh) {
          const size_t step_size = step_size_outer;
          const Complex omega = std::exp(-2.0 * J * M_PI / (double)step_size);
          auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<class fft_calculations_small_size>(
              sycl::range<1>(step_size / 2), [=](sycl::id<1> idx) {
                size_t index_even = start + idx[0];
                size_t index_odd = start + idx[0] + step_size / 2;
                Complex temp = acc[index_even];
                acc[index_even] =
                    temp + std::pow(omega, idx[0]) * acc[index_odd];
                acc[index_odd] =
                    temp - std::pow(omega, idx[0]) * acc[index_odd];
              });
        });
      }
    }

    queue.wait();
  }
}

int main(int argc, char *argv[]) {

  std::vector<Complex> array;
  size_t size = std::pow(2, 29);

  std::ifstream inputFile("../../../TEST_DATA/FFT_test_vector",
                          std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Error: Failed to open file for reading.\n";
    return -1;
  }

  inputFile.seekg(0, std::ios::end);
  std::streampos fileSize = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  size = fileSize / sizeof(Complex);

  array = std::vector<Complex>(size);
  inputFile.read(reinterpret_cast<char *>(array.data()), fileSize);

  inputFile.close();

  bit_reverse_indices(&array, size);

  auto start = std::chrono::high_resolution_clock::now();

  fft(&array, size);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << "FFT zostało wykonane w czasie: " << duration.count() << " sekund\n";

  /* std::ofstream outputFile("output.dat", std::ios::binary); */
  /* if (!outputFile.is_open()) { */
  /*   std::cerr << "Error: Failed to open file for writing.\n"; */
  /*   return -1; */
  /* } */

  /* outputFile.write(reinterpret_cast<const char *>(array.data()), */
  /*                  array.size() * sizeof(Complex)); */

  /* outputFile.close(); */

  return 0;
}
