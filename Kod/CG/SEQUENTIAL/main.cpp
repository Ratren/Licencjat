#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

using vec = std::vector<double>;
using mat = std::vector<vec>;

mat gen_random_matrix(uint32_t size, uint32_t seed) {
  mat out(size, vec(size));

  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(seed);
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j < size; ++j) {
      out[i][j] = dis(gen);
    }
  }

  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j < size; ++j) {
      out[i][j] = 0.5 * (out[i][j] + out[j][i]);
    }
  }

  for (uint32_t i = 0; i < size; ++i) {
    out[i][i] += size * size;
  }

  return out;
}

vec gen_random_vector(uint32_t size, uint32_t seed) {
  vec out(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(seed);
  std::uniform_real_distribution<double> dis(0.0, 40.0);

  for (uint32_t i = 0; i < size; ++i) {
    /* out[i] = dis(gen); */
    out[i] = 0.0127734547 * i+1;
  }

  return out;
}

double inner_product(const vec &V, const vec &V2) {
  return std::inner_product(V.begin(), V.end(), V2.begin(), 0.0);
}

vec matrix_vector_multiply(const mat &A, const vec &V) {
  uint32_t size = A.size();
  vec out(size);
  for (int i = 0; i < size; ++i) {
    out[i] = inner_product(A[i], V);
  }
  return out;
}

double norm(const vec &V) {
  return sqrt(inner_product(V, V));
}

vec vector_combination(double a, const vec &V, double b, const vec &V2) {
  uint32_t size = V.size();
  vec combination(size);
  for (int i=0; i<size; ++i)
    combination[i] = a * V[i] + b * V2[i];
  return combination;
}

vec conjugate_gradient(const mat &A, const vec &B) {
  double tolerance = 1.0e-27;
  
  uint32_t size = A.size();
  vec X(size, 0.0);

  vec residual = B;
  vec search_dir = residual;
  
  double old_resid_norm = norm(residual);

  while (old_resid_norm > tolerance) {
    vec A_search_dir = matrix_vector_multiply(A, search_dir);

    double alpha = old_resid_norm*old_resid_norm / inner_product(search_dir, A_search_dir);
    X = vector_combination(1.0, X, alpha, search_dir); 
    residual = vector_combination(1.0, residual, -alpha, A_search_dir);

    double new_resid_norm = norm(residual);
   
    double pow = std::pow(new_resid_norm/old_resid_norm, 2);
    for (uint32_t i=0; i<size; ++i) {
      search_dir[i] = residual[i] + pow * search_dir[i];
    }
    old_resid_norm = new_resid_norm;
  }

  return X;
}

int main() {

  uint32_t size = 12 * 3200;
  uint32_t seed = 23453745;

  mat A = gen_random_matrix(size, seed);
  vec B = gen_random_vector(size, seed);
  vec X; 
  auto start = std::chrono::high_resolution_clock::now();
  for (int i=0; i<10; ++i) {
    X = conjugate_gradient(A, B);
  }
  /* X = conjugate_gradient(A, B); */
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  double seconds = duration.count();

  std::cout << "Zajelo: " << seconds << " sekund" << std::endl;

  

  return 0;
}
