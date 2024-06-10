#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <device_selector.hpp>
#include <fstream>
#include <oneapi/dpl/random>
#include <queue.hpp>

void generate_vec(double *array, size_t size, uint32_t seed) {

  sycl::queue queue(sycl::cpu_selector_v);

  sycl::buffer<double, 1> buf(array, sycl::range<1>(size));

  queue.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class generate_random>(
        sycl::range<1>(size), [=](sycl::id<1> idx) {
          oneapi::dpl::minstd_rand engine(seed, idx);
          oneapi::dpl::uniform_real_distribution<double> distr(0.0, 40.0);
          acc[idx] = distr(engine);
        });
  });

  queue.wait_and_throw();
}

void generate_mat(double *array, size_t size, uint32_t seed) {
  
  sycl::queue queue(sycl::cpu_selector_v);

  sycl::buffer<double, 1> buf(array, sycl::range<1>(size*size));

  queue.submit([&](sycl::handler &cgh) {
    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class generate_random_mat>(
        sycl::range<1>(size*size), [=](sycl::id<1> idx) {
          oneapi::dpl::minstd_rand engine(seed, idx);
          oneapi::dpl::uniform_real_distribution<double> distr;
          acc[idx] = distr(engine);
        });
  });

  queue.wait_and_throw();
  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j < size; ++j) {
      array[size*i+j] = 0.5 * (array[size*i+j] + array[size*j+i]); 
    }
  }

  for (uint32_t i=0; i<size; ++i) {
    array[i*size+i] += size * size;
  }
}

int main(int argc, char *argv[]) {

  if (argc != 4) {
    std::cout << "Uzycie: " << argv[0]
              << " <mnoznik-12> <ziarno-macierzy> <ziarno-wektora>\n";
    return 0;
  }

  size_t size = 12 * atoi(argv[1]);
  uint32_t seed_m = std::stoll(argv[2]);
  uint32_t seed_v = std::stoll(argv[3]);
  double *matrix = new double[size * size];
  double *vector = new double[size];
  generate_mat(matrix, size, seed_m);
  generate_vec(vector, size, seed_v);

  std::ofstream matrixFile("../../../TEST_DATA/CG_test_matrix", std::ios::binary);
  if (!matrixFile.is_open()) {
    std::cerr << "Error: Failed to open matrix file for writing." << std::endl;
    return -1;
  }

  matrixFile.write(reinterpret_cast<const char *>(matrix),
                   size * size * sizeof(double));
  matrixFile.close();

  std::ofstream vectorFile("../../../TEST_DATA/CG_test_vector", std::ios::binary);
  if (!vectorFile.is_open()) {
    std::cerr << "Error: Failed to open vector file for writing." << std::endl;
    return -1;
  }

  vectorFile.write(reinterpret_cast<const char *>(vector),
                   size * sizeof(double));
  vectorFile.close();

  delete[] matrix;
  delete[] vector;

  return 0;
}
