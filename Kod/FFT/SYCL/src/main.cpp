#include <CL/sycl.hpp>
#include <access/access.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <oneapi/dpl/random>
#include <queue.hpp>
#include <time.h>
#include <vector>
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

  std::vector<Complex> omega_pows(size);
  sycl::buffer<Complex, 1> buf_omega_pows(omega_pows.data(),
                                            sycl::range<1>(size));
  for (int i = 1; i <= num_bits; ++i) {
    int step_size = 1 << i;
    Complex omega = std::exp(-2.0 * J * M_PI / (double)step_size);
    omega_pows[0] = 1;
    for (int j = 1; j < step_size / 2; ++j) {
      omega_pows[j] = omega_pows[j - 1] * omega;
    }

    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
      auto omega_pows_acc =
          buf_omega_pows.get_access<sycl::access_mode::read>(cgh);
      cgh.parallel_for<class fft_calculations>(
          sycl::range<1>(size), [=](sycl::id<1> idx) {
            int r = idx[0] % step_size;
            if (r < step_size / 2) {
              Complex t = omega_pows_acc[r] * acc[idx[0] + step_size / 2];
              Complex x = acc[idx[0]];
              acc[idx[0]] = x + t;
              acc[idx[0] + step_size / 2] = x - t;
            }
          });
    });
    queue.wait();
    std::cout << "Step " << i << ": " << input_array->data()[0] << '\n';
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

  std::cout << "FFT zostało wykonane w czasie: " << duration.count()
            << " sekund\n";

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
