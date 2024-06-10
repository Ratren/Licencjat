#include <access/access.hpp>
#include <device_selector.hpp>
#include <fstream>
#include <functional.hpp>
#include <iostream>
#include <range.hpp>
#include <sycl/sycl.hpp>

int read_data_from_files(double *&matrix, double *&vector) {
  int size;

  std::ifstream vectorFile("../../../TEST_DATA/CG_test_vector",
                           std::ios::binary);
  if (!vectorFile.is_open()) {
    std::cerr << "Error: Failed to open vector file for reading.\n";
    return -1;
  }

  vectorFile.seekg(0, std::ios::end);
  std::streampos fileSize = vectorFile.tellg();
  vectorFile.seekg(0, std::ios::beg);

  size = fileSize / sizeof(double);

  vector = new double[size];
  matrix = new double[size * size];

  vectorFile.read(reinterpret_cast<char *>(vector), fileSize);
  vectorFile.close();

  std::ifstream matrixFile("../../../TEST_DATA/CG_test_matrix",
                           std::ios::binary);
  if (!matrixFile.is_open()) {
    std::cerr << "Error: Failed to open matrix file for reading.\n";
    return -1;
  }

  matrixFile.seekg(0, std::ios::end);
  std::streampos matrixFileSize = matrixFile.tellg();
  matrixFile.seekg(0, std::ios::beg);

  matrixFile.read(reinterpret_cast<char *>(matrix), matrixFileSize);
  matrixFile.close();

  return size;
}

double norm(sycl::queue &queue, sycl::buffer<double, 1> &V_buf, int size) {
  double inner_prod = 0.0;
  sycl::buffer<double, 1> inner_prod_buf(&inner_prod, sycl::range<1>(1));
  queue.submit([&](sycl::handler &h) {
    auto V = V_buf.get_access<sycl::access::mode::read>(h);
    auto inner_prod_acc =
        inner_prod_buf.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(size),
                   [=](sycl::id<1> i) {
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
          atomic_inner_prod(inner_prod_acc[0]);
      atomic_inner_prod.fetch_add(V[i] * V[i]);
    });
  });
  queue.wait();
  return std::sqrt(inner_prod);
}

double inner_product(sycl::queue &queue, sycl::buffer<double, 1> &V1_buf,
                     sycl::buffer<double, 1> &V2_buf, int size) {
  double inner_prod = 0.0;
  sycl::buffer<double, 1> inner_prod_buf(&inner_prod, sycl::range<1>(1));
  queue.submit([&](sycl::handler &h) {
    auto V1 = V1_buf.get_access<sycl::access::mode::read>(h);
    auto V2 = V2_buf.get_access<sycl::access::mode::read>(h);
    auto inner_prod_acc =
        inner_prod_buf.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(size),
                   [=](sycl::id<1> i) {
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
          atomic_inner_prod(inner_prod_acc[0]);
      atomic_inner_prod.fetch_add(V1[i] * V2[i]);
    });
  });
  queue.wait();
  return inner_prod;
}

void matrix_vector_multiply(sycl::queue &queue,
                            sycl::buffer<double, 1> &mat_buf,
                            sycl::buffer<double, 1> &vec_buf,
                            sycl::buffer<double, 1> &output_buf, int size) {
  queue.submit([&](sycl::handler &h) {
    auto mat = mat_buf.get_access<sycl::access::mode::read>(h);
    auto vec = vec_buf.get_access<sycl::access::mode::read>(h);
    auto output = output_buf.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
      double sum = 0.0;
      for (int j = 0; j < size; ++j) {
        sum += mat[i * size + j] * vec[j];
      }
      output[i] = sum;
    });
  });
  queue.wait();
}

void vector_combination(sycl::queue &queue, sycl::buffer<double, 1> &vec1_buf,
                        double mult, sycl::buffer<double, 1> &vec2_buf,
                        int size) {
  queue.submit([&](sycl::handler &h) {
    auto vec1 = vec1_buf.get_access<sycl::access::mode::read_write>(h);
    auto vec2 = vec2_buf.get_access<sycl::access::mode::read>(h);

    h.parallel_for(sycl::range<1>(size),
                   [=](sycl::id<1> i) { vec1[i] += vec2[i] * mult; });
  });
  queue.wait();
}

void conjugate_gradient(const double *A, const double *B, double *X, int size,
                        sycl::queue &queue) {
  double tolerance = 1.0e-27;

  std::memset(X, 0.0, size * sizeof(double));

  sycl::buffer<double, 1> A_buf{A, sycl::range<1>(size * size)};
  sycl::buffer<double, 1> X_buf{X, sycl::range<1>(size)};
  sycl::buffer<double, 1> B_buf{B, sycl::range<1>(size)};

  double *residual_a = new double[size];
  double *search_dir_a = new double[size];
  double *A_search_dir_a = new double[size];

  sycl::buffer<double, 1> residual{residual_a,
                                   sycl::range<1>(size)}; // set it to B
  sycl::buffer<double, 1> search_dir{search_dir_a,
                                     sycl::range<1>(size)}; // set it to B

  queue.submit([&](sycl::handler &h) {
    auto B_acc = B_buf.get_access<sycl::access::mode::read>(h);
    auto residual_acc = residual.get_access<sycl::access::mode::write>(h);
    auto search_dir_acc = search_dir.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
      residual_acc[i] = B_acc[i];
      search_dir_acc[i] = B_acc[i];
    });
  });
  queue.wait();

  sycl::buffer<double, 1> A_search_dir{A_search_dir_a, sycl::range<1>(size)};

  double old_resid_norm = norm(queue, residual, size);
  double alpha;
  double new_resid_norm;
  double pow;

  while (old_resid_norm > tolerance) {
    matrix_vector_multiply(queue, A_buf, search_dir, A_search_dir, size);


    alpha = old_resid_norm * old_resid_norm /
            inner_product(queue, search_dir, A_search_dir, size);
    vector_combination(queue, X_buf, alpha, search_dir, size);
    vector_combination(queue, residual, -alpha, A_search_dir, size);

    new_resid_norm = norm(queue, residual, size);

    pow = std::pow(new_resid_norm / old_resid_norm, 2);
    queue.submit([&](sycl::handler &h) {  
      auto residual_acc = residual.get_access<sycl::access::mode::write>(h);
      auto search_dir_acc = search_dir.get_access<sycl::access::mode::read_write>(h);

      h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
        search_dir_acc[i] = residual_acc[i] + pow * search_dir_acc[i];
      });
    });
    queue.wait();
    old_resid_norm = new_resid_norm;
  }

  /* delete[] residual_a; */
  /* delete[] search_dir_a; */
  /* delete[] A_search_dir_a; */

}

int main() {

  double *A;
  double *B;

  int size = read_data_from_files(A, B);
  double *X = new double[size];

  sycl::queue queue{sycl::cpu_selector_v};

  for (int i=0 ; i<100; ++i)
    conjugate_gradient(A, B, X, size, queue);

  /* delete[] A; */
  /* delete[] B; */
  /* delete[] X; */

  return 0;
}
