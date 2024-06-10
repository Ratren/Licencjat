#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <math.h>
#include <mpi.h>

void print(std::string title, double *vec, int size) {
  std::cout << title << '\n';

  for (int i = 0; i < size; ++i) {
    std::cout << vec[i] << '\t';
  }
  std::cout << '\n';
}

int read_data_from_files(double *&matrix, double *&vector, int rank, int root,
                         int num_procs) {
  int size;
  double *fullArray = nullptr;

  if (rank == root) {
    std::ifstream vectorFile("../../TEST_DATA/CG_test_vector",
                             std::ios::binary);
    if (!vectorFile.is_open()) {
      std::cerr << "Error: Failed to open vector file for reading.\n";
      return -1;
    }

    vectorFile.seekg(0, std::ios::end);
    std::streampos fileSize = vectorFile.tellg();
    vectorFile.seekg(0, std::ios::beg);

    size = fileSize / sizeof(double);
    fullArray = new double[size];
    vectorFile.read(reinterpret_cast<char *>(fullArray), fileSize);
    vectorFile.close();
  }

  MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  vector = new double[size / num_procs];
  matrix = new double[size * size / num_procs];
  MPI_Scatter(fullArray, size / num_procs, MPI_DOUBLE, vector, size / num_procs,
              MPI_DOUBLE, root, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == root) {
    delete[] fullArray;

    std::ifstream matrixFile("../../TEST_DATA/CG_test_matrix",
                             std::ios::binary);
    if (!matrixFile.is_open()) {
      std::cerr << "Error: Failed to open matrix file for reading.\n";
      return -1;
    }

    fullArray = new double[size * size];
    matrixFile.read(reinterpret_cast<char *>(fullArray),
                    size * size * sizeof(double));
    matrixFile.close();
  }

  MPI_Scatter(fullArray, size * size / num_procs, MPI_DOUBLE, matrix,
              size * size / num_procs, MPI_DOUBLE, root, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == root)
    delete[] fullArray;

  return size;
}

void matrix_vector_multiply(double *matrix, double *input_vector,
                            double *output_vector, int size, int num_procs) {
  int num_rows = size / num_procs;
  for (int i = 0; i < num_rows; ++i) {
    double product = 0.0;
    for (int j = 0; j < size; ++j) {
      product += matrix[i * size + j] * input_vector[j];
    }
    output_vector[i] = product;
  }
}

// combines vector1 and vector2 into vector1 taking vector1*mult1 +
// vector2*mult2
void vector_combination(double mult1, double *vector1, double mult2,
                        double *vector2, int size) {
  for (int i = 0; i < size; ++i)
    vector1[i] = mult1 * vector1[i] + mult2 * vector2[i];
}

double norm(double *vec, int size) {
  double sum_of_squares_local = 0;
  double sum_of_squares = 0;
  for (int i = 0; i < size; ++i) {
    sum_of_squares_local += vec[i] * vec[i];
  }
  MPI_Allreduce(&sum_of_squares_local, &sum_of_squares, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return sqrt(sum_of_squares);
}

double inner_product(double *vec1, double *vec2, int size, int rank) {
  double local_prod = 0.0, prod = 0.0;
  for (int i = 0; i < size; ++i) {
    local_prod += vec1[i + size * rank] * vec2[i];
  }
  MPI_Allreduce(&local_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return prod;
}

void conjugate_gradient(double *A, double *B, double *X, int size, int rank,
                        int num_procs, int root) {

  double *A_search_dir = new double[size / num_procs];

  double *residual = new double[size / num_procs];
  std::copy(B, B + (size / num_procs), residual);

  double *search_dir = new double[size];
  MPI_Allgather(B, size / num_procs, MPI_DOUBLE, search_dir, size / num_procs,
                MPI_DOUBLE, MPI_COMM_WORLD);

  double *search_dir_local = new double[size / num_procs];

  MPI_Barrier(MPI_COMM_WORLD);
  double tolerance = 1.0e-27;

  int local_size = size / num_procs;
  double old_resid_norm = norm(residual, local_size);

  while (old_resid_norm > tolerance) {
    matrix_vector_multiply(A, search_dir, A_search_dir, size, num_procs);


    double alpha = old_resid_norm * old_resid_norm /
                   inner_product(search_dir, A_search_dir, local_size, rank);
    vector_combination(1.0, X, alpha, search_dir + (rank * local_size),
                       local_size);
    vector_combination(1.0, residual, -alpha, A_search_dir, local_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double new_resid_norm = norm(residual, local_size);

    double pow = std::pow(new_resid_norm / old_resid_norm, 2);
    for (int i = 0; i < local_size; ++i) {
      search_dir_local[i] =
          residual[i] + pow * search_dir[(rank * local_size) + i];
    }
    MPI_Allgather(search_dir_local, size / num_procs, MPI_DOUBLE, search_dir,
                  size / num_procs, MPI_DOUBLE, MPI_COMM_WORLD);
    old_resid_norm = new_resid_norm;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  delete[] A_search_dir;
  delete[] residual;
  delete[] search_dir;
  delete[] search_dir_local;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, num_procs, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  double *A, *B, *X;
  int size = read_data_from_files(A, B, rank, root, num_procs);
  X = new double[size / num_procs];
  std::memset(X, 0.0, (size / num_procs) * sizeof(double));

  double t, t_start, t_stop;

  if (rank==root) t_start = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i=0; i<100; ++i) {  
    std::memset(X, 0.0, (size / num_procs) * sizeof(double));
    conjugate_gradient(A, B, X, size, rank, num_procs, root);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == root) {
    t_stop = MPI_Wtime();
    t = t_stop-t_start;
    std::cout << "Czas_dzialania_conjugate_gradient: ";
    printf("%1.2f\n", t);
  }

/*
  double *fullX, *result;
  fullX = new double[size];
  result = new double[size / num_procs];
  MPI_Allgather(X, size / num_procs, MPI_DOUBLE, fullX, size / num_procs,
                MPI_DOUBLE, MPI_COMM_WORLD);

  matrix_vector_multiply(A, fullX, result, size, num_procs);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank==root)
    std::cout << "\n\n";

  for (int i = 0; i < num_procs; ++i) {
    if (rank == i) {
      for (int j = 0; j < size / num_procs; ++j) {
        std::cout << B[j] << '\t';
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (rank == root)
    std::cout << "\n\n\n\n\n\n";

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < num_procs; ++i) {
    if (rank == i) {
      for (int j = 0; j < size / num_procs; ++j) {
        std::cout << result[j] << '\t';
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  if (rank == root)
    std::cout << '\n';
*/
  delete[] A;
  delete[] B;
  delete[] X;
//  delete[] fullX;
//  delete[] result;
  MPI_Finalize();

  return 0;
}
