#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <device_selector.hpp>
#include <oneapi/dpl/random>
#include <queue.hpp>
#include <vector>
#include <fstream>
#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

std::vector<Complex> generate_input_array(size_t size, uint32_t seed) {

  size_t chunkSize = size / 8;
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

int main(int argc, char* argv[]) {

  if (argc != 3) {
    std::cerr << "Uzycie: " << argv[0] << "<potega_2> <ziarno>"; 
  }

  size_t size = std::pow(2, std::stoi(argv[1]));
  uint32_t seed = std::stoll(argv[2]);
  std::vector<Complex> array = generate_input_array(size, seed);

  std::ofstream outputFile("../../../TEST_DATA/FFT_test_vector", std::ios::binary);
  if (!outputFile.is_open()) {
    std::cerr << "Error: Failed to open file for writing." << std::endl;
    return -1;
  }

  outputFile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(Complex));
  outputFile.close();
  
  return 0;
}
