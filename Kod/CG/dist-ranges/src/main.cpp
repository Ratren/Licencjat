#include <dr/mhp.hpp>
#include <fmt/core.h>
#include <mpi.h>
#include <mpio.h>
#include <oneapi/dpl/random>

namespace mhp = dr::mhp;

template <dr::distributed_range X> void gen_random_matrix(X &&mat, int size) {

  mhp::distributed_vector<int> idx(size * size);
  mhp::iota(idx, 0);

  mhp::for_each(mhp::views::zip(mat, idx), [=](auto &&elem) {
    auto &&[val, i] = elem;
    oneapi::dpl::minstd_rand engine(283457, i);
    oneapi::dpl::uniform_real_distribution<double> distr(0.0, 1.0);
    val = distr(engine);
  });

  if (mhp::rank() == 0) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        mat[i * size + j] = 0.5 * (mat[i * size + j] + mat[j * size + i]);
      }
    }

    for (int i = 0; i < size; ++i) {
      mat[i + i * size] = mat[i + i * size] + size * size;
    }
  }
}

template <dr::distributed_range X> void gen_random_vector(X &&vec, int size) {

  mhp::distributed_vector<int> idx(size);
  mhp::iota(idx, 0);

  mhp::for_each(mhp::views::zip(vec, idx), [=](auto &&elem) {
    auto &&[val, i] = elem;
    oneapi::dpl::minstd_rand engine(283457, i);
    oneapi::dpl::uniform_real_distribution<double> distr(0.0, 40.0);
    val = distr(engine);
  });
}

template <dr::distributed_range X>
void vector_combination(X &&vec, std::vector<double> &vec2, double mult) {
  auto zipped = dr::mhp::views::zip(vec, vec2);
  mhp::for_each(zipped, [=](auto &&elem) {
    auto &&[a, b] = elem;
    a = a + b * mult;
  });
}

template <dr::distributed_range X, dr::distributed_range Y>
void vector_combination(X &&vec, Y &&vec2, double mult) {
  auto zipped = dr::mhp::views::zip(vec, vec2);
  mhp::for_each(zipped, [=](auto &&elem) {
    auto &&[a, b] = elem;
    a = a + b * mult;
  });
}

template <dr::distributed_range X, dr::distributed_range Y>
double inner_product(X &&x, Y &&y) {
  auto zipped =
      dr::mhp::views::zip(x, y) | dr::mhp::views::transform([](auto &&elem) {
        auto &&[a, b] = elem;
        return a * b;
      });

  return mhp::reduce(zipped);
}

template <dr::distributed_range X>
double inner_product(X &&x, std::vector<double> &y) {
  auto zipped =
      dr::mhp::views::zip(x, y) | dr::mhp::views::transform([](auto &&elem) {
        auto &&[a, b] = elem;
        return a * b;
      });

  return mhp::reduce(zipped);
}

template <dr::distributed_range X> double norm(X &&vec) {
  double inner_prod = inner_product(vec, vec);
  return std::sqrt(inner_prod);
}

template <dr::distributed_range X, dr::distributed_range Y>
void matrix_vector_multiply(X &&mat, std::vector<double> &vec, Y &&result,
                            int size) {
  auto segment = mat.segments()[mhp::rank()];
  int rows_in_segment = size / mhp::nprocs();
  std::vector<double> result_vec(rows_in_segment, 0);
  for (int i = 0; i < rows_in_segment; ++i) {
    for (int j=0; j < size; ++j) {
      result_vec[i] += segment[size * i + j] * vec[j];
    }
  }

  segment = result.segments()[mhp::rank()];
  for (int i = 0; i < rows_in_segment; ++i) {
    segment[i] = result_vec[i];
  }

  mhp::barrier();
}

template <dr::distributed_range X, dr::distributed_range Y,
          dr::distributed_range Z>
void conjugate_gradient(X &&A, Y &&B, Z &&_X) {
  int size = B.size();
  double tolerance = 1.0e-27;

  mhp::fill(_X, 0.0);

  mhp::distributed_vector<double> residual(size);
  mhp::copy(B, residual.begin());
  std::vector<double> search_dir(size);
  std::copy(B.begin(), B.end(), search_dir.begin());
  mhp::distributed_vector<double> A_search_dir(size);
  double old_resid_norm = norm(residual);
  double new_resid_norm, pow, alpha;

  mhp::barrier();

  while (old_resid_norm > tolerance) {
    matrix_vector_multiply(A, search_dir, A_search_dir, size);

    alpha = old_resid_norm * old_resid_norm /
            inner_product(A_search_dir, search_dir);
    vector_combination(_X, search_dir, alpha);
    vector_combination(residual, A_search_dir, -alpha);

    new_resid_norm = norm(residual);

    pow = std::pow(new_resid_norm / old_resid_norm, 2);

    for (int i = 0; i < size; ++i) {
      search_dir[i] = residual[i] + pow * search_dir[i];
    }
    old_resid_norm = new_resid_norm;
  }
}

int main(int argc, char **argv) {

  mhp::init(sycl::cpu_selector_v);

  // READ MATRIX
  MPI_Status status;
  MPI_File fh;
  int file_open_error =
      MPI_File_open(MPI_COMM_WORLD, "./TEST_DATA/CG_test_matrix",
                    MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (file_open_error != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, file_open_error);
  }

  MPI_Offset total_number_of_bytes;
  int local_number_of_bytes;
  MPI_File_get_size(fh, &total_number_of_bytes);
  local_number_of_bytes = total_number_of_bytes / mhp::nprocs();
  MPI_File_seek(fh, local_number_of_bytes * mhp::rank(), MPI_SEEK_SET);

  mhp::barrier();

  int total_data_size = total_number_of_bytes / sizeof(double);
  mhp::distributed_vector<double> A(total_data_size);

  int local_data_size = local_number_of_bytes / sizeof(double);
  double *local_data = new double[local_data_size];
  MPI_File_read(fh, local_data, local_number_of_bytes, MPI_BYTE, &status);
  MPI_File_close(&fh);

  auto segment = A.segments()[mhp::rank()];

  for (int i = 0; i < local_data_size; ++i) {
    segment[i] = local_data[i];
  }

  delete[] local_data;

  // READ VECTOR
  file_open_error =
      MPI_File_open(MPI_COMM_WORLD, "./TEST_DATA/CG_test_vector",
                    MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (file_open_error != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, file_open_error);
  }

  MPI_File_get_size(fh, &total_number_of_bytes);
  local_number_of_bytes = total_number_of_bytes / mhp::nprocs();
  MPI_File_seek(fh, local_number_of_bytes * mhp::rank(), MPI_SEEK_SET);

  mhp::barrier();

  total_data_size = total_number_of_bytes / sizeof(double);
  mhp::distributed_vector<double> B(total_data_size);
  local_data_size = local_number_of_bytes / sizeof(double);
  local_data = new double[local_data_size];
  MPI_File_read(fh, local_data, local_number_of_bytes, MPI_BYTE, &status);
  MPI_File_close(&fh);

  segment = B.segments()[mhp::rank()];

  for (int i = 0; i < local_data_size; ++i) {
    segment[i] = local_data[i];
  }

  delete[] local_data;

  mhp::barrier();

  int size = total_data_size;
  mhp::distributed_vector<double> X(size);

  double t, t_start, t_stop;

  if (mhp::rank() == 0) {
    t_start = MPI_Wtime();
  }

  conjugate_gradient(A, B, X);

  if (mhp::rank() == 0) {
    t_stop = MPI_Wtime();
    t = t_stop - t_start;
    std::cout << "Zajelo: ";
    printf("%1.2f\n", t);
  }

  mhp::finalize();

  return 0;
}
