#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <random>
#include <boost/multiprecision/cpp_bin_float.hpp>

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val = std::pow(val, 20); });
  return std::sin(std::reduce(x.begin(), x.end()));
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, num_procs, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  int num_dimensions = 100, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  std::vector<double> x(num_dimensions-1);
  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<double> dis(a, b);
  gen.seed(time(NULL) + rank);


  for (int i = 0; i < num_iterations / num_procs; ++i) {
    std::for_each(x.begin(), x.end(), [&](double &val) { val = dis(gen); });
    count_under += f(x) > dis(gen);
  }

  int global_count = 0;
  MPI_Reduce(&count_under, &global_count, 1, MPI_INT, MPI_SUM, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    boost::multiprecision::cpp_bin_float_quad V = std::pow(b - a, num_dimensions);
    boost::multiprecision::cpp_bin_float_quad result = V * global_count / num_iterations;

    std::cout << result << " " << global_count << " ";
  }

  MPI_Finalize();

  return 0;
}
