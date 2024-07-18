#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

using vec = std::vector<double>;

int read_data_from_files(vec &matrix, vec &vector) {
  int size;

  std::ifstream vectorFile("./TEST_DATA/CG_test_vector", std::ios::binary);
  if (!vectorFile.is_open()) {
    std::cerr << "Error: Failed to open vector file for reading.\n";
    return -1;
  }

  vectorFile.seekg(0, std::ios::end);
  std::streampos fileSize = vectorFile.tellg();
  vectorFile.seekg(0, std::ios::beg);

  size = fileSize / sizeof(double);

  vector.resize(size);
  matrix.resize(size * size);

  vectorFile.read(reinterpret_cast<char *>(vector.data()), fileSize);
  vectorFile.close();

  std::ifstream matrixFile("./TEST_DATA/CG_test_matrix", std::ios::binary);
  if (!matrixFile.is_open()) {
    std::cerr << "Error: Failed to open matrix file for reading.\n";
    return -1;
  }

  matrixFile.seekg(0, std::ios::end);
  std::streampos matrixFileSize = matrixFile.tellg();
  matrixFile.seekg(0, std::ios::beg);

  matrixFile.read(reinterpret_cast<char *>(matrix.data()), matrixFileSize);
  matrixFile.close();

  return size;
}

double inner_product(const vec &V, const vec &V2) {
  return std::inner_product(V.begin(), V.end(), V2.begin(), 0.0);
}

vec matrix_vector_multiply(const vec &A, const vec &V) {
  uint32_t size = V.size();
  vec out(size);
  for (int i = 0; i < size; ++i) {
    out[i] = std::inner_product(V.begin(), V.end(), A.begin() + size * i, 0.0);
  }
  return out;
}

double norm(const vec &V) { return sqrt(inner_product(V, V)); }

vec vector_combination(double a, const vec &V, double b, const vec &V2) {
  uint32_t size = V.size();
  vec combination(size);
  for (int i = 0; i < size; ++i)
    combination[i] = a * V[i] + b * V2[i];
  return combination;
}

vec conjugate_gradient(const vec &A, const vec &B) {
  double tolerance = 1.0e-27;

  uint32_t size = B.size();
  vec X(size, 0.0);

  vec residual = B;
  vec search_dir = residual;

  double old_resid_norm = norm(residual);

  while (old_resid_norm > tolerance) {
    vec A_search_dir = matrix_vector_multiply(A, search_dir);

    double alpha = old_resid_norm * old_resid_norm /
                   inner_product(search_dir, A_search_dir);
    X = vector_combination(1.0, X, alpha, search_dir);
    residual = vector_combination(1.0, residual, -alpha, A_search_dir);

    double new_resid_norm = norm(residual);

    double pow = std::pow(new_resid_norm / old_resid_norm, 2);
    for (uint32_t i = 0; i < size; ++i) {
      search_dir[i] = residual[i] + pow * search_dir[i];
    }
    old_resid_norm = new_resid_norm;
  }

  return X;
}

int main() {

  vec A;
  vec B;
  int size = read_data_from_files(A, B);
  vec X;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    X = conjugate_gradient(A, B);
  }
  /* X = conjugate_gradient(A, B); */
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  double seconds = duration.count();

  std::cout << "Zajelo: " << seconds << " sekund" << std::endl;

  return 0;
}
