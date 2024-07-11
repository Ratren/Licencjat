#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <random>

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val *= std::pow(val, 5); });
  return std::exp(std::reduce(x.begin(), x.end()));
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, num_procs, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  int num_arguments = 120, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  std::vector<double> x(num_arguments);
  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<double> dis(a, b);
  std::uniform_real_distribution<double> dis_2(-1.0, 100000000.0);
  gen.seed(time(NULL) + rank);


  for (int i = 0; i < num_iterations / num_procs; ++i) {
    std::for_each(x.begin(), x.end(), [&](double &val) { val = dis(gen); });
    count_under += f(x) > dis_2(gen);
  }

  int global_count = 0;
  MPI_Reduce(&count_under, &global_count, 1, MPI_INT, MPI_SUM, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    double V = std::pow(b - a, num_arguments);
    double result = V * count_under / num_iterations;

    std::cout << result << " " << global_count << " ";
  }

  MPI_Finalize();

  return 0;
}
