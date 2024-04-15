#include <complex>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>

#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

void bit_reverse_indices(size_t size, unsigned long num_bits,
                         Complex *input_array) {
  Complex *temp_array = new Complex[size];
  std::copy(input_array, input_array + size, temp_array);
  unsigned long tableSize = 1 << num_bits;
  for (unsigned long i = 0; i < tableSize; ++i) {
    unsigned long reversed = 0;
    for (unsigned long j = 0; j < num_bits; ++j) {
      if (i & (1 << j)) {
        reversed |= (1 << (num_bits - 1 - j));
      }
    }
    input_array[i] = temp_array[reversed];
  }
}

void random_values_into_array(Complex *array, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (size_t i = 0; i < size; ++i) {
    array[i] = Complex(dis(gen), 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void init_random_values(Complex *global_array, size_t size, int rank, int root,
                        int num_procs) {

  Complex *local_array = nullptr;

  size_t local_size;

  if (rank != num_procs - 1) {
    local_size = size / num_procs;
  } else {
    local_size = (size / num_procs) + size % num_procs;
  }

  local_array = new Complex[local_size];

  int rcvcnts[num_procs], displs[num_procs];

  for (int i = 0; i < num_procs - 1; ++i) {
    rcvcnts[i] = size / num_procs;
    displs[i] = i * (size / num_procs);
  }
  rcvcnts[num_procs - 1] = (size / num_procs) + size % num_procs;
  displs[num_procs - 1] = (num_procs - 1) * (size / num_procs);

  random_values_into_array(local_array, local_size);

  MPI_Gatherv(local_array, local_size, MPI_DOUBLE_COMPLEX, global_array,
              rcvcnts, displs, MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);

  delete[] local_array;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, num_procs, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  size_t num_bits = 29;
  size_t size = std::pow(2, num_bits);
  Complex *global_array = nullptr;

  if (rank == root) {
    global_array = new Complex[size];
  }

  init_random_values(global_array, size, rank, root, num_procs);

  if (rank == 0) {
    std::cout << global_array[0] << " " << global_array[size-1] << '\n';
  }

  for (size_t step = 1; step <= num_bits; ++step) {
    size_t step_size = 1 << step;
    Complex omega =
        std::pow(std::exp(1), (2 * M_PI * J) / static_cast<double>(step_size));
    size_t num_chunks = size / step_size;

    if (rank == root) {
      std::cout << step << " " << num_chunks << '\n';
    }
    if (num_chunks >= 512) {

      Complex *local_array = new Complex[step_size];
      int num_regular_waves = num_chunks / num_procs;
      for (int i = 0; i < num_regular_waves; ++i) {
        MPI_Scatter(global_array + i * num_procs * step_size, step_size,
                    MPI_DOUBLE_COMPLEX, local_array, step_size,
                    MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);

        for (size_t i = 0; i < step_size / 2; i++) {
          size_t index_odd = i + step_size / 2;
          Complex angle = std::pow(omega, i);
          Complex temp = local_array[i];
          local_array[i] = temp + angle * local_array[index_odd];
          local_array[index_odd] = temp - angle * local_array[index_odd];
        }

        MPI_Gather(local_array, step_size, MPI_DOUBLE_COMPLEX,
                   global_array + i * num_procs * step_size, step_size,
                   MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
      }
      int leftovers = num_chunks % num_procs;
      int sendcnts[num_procs], displs[num_procs];
      for (int i = 0; i < num_procs; ++i) {
        if (i < leftovers) {
          sendcnts[i] = step_size;
          displs[i] =
              (num_regular_waves * num_procs * step_size + i * step_size);
        } else {
          sendcnts[i] = 0;
          displs[i] = 0;
        }
      }
      MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                   local_array, step_size, MPI_DOUBLE_COMPLEX, root,
                   MPI_COMM_WORLD);
      for (size_t i = 0; i < step_size / 2; i++) {
        size_t index_odd = i + step_size / 2;
        Complex angle = std::pow(omega, i);
        Complex temp = local_array[i];
        local_array[i] = temp + angle * local_array[index_odd];
        local_array[index_odd] = temp - angle * local_array[index_odd];
      }
      MPI_Gatherv(local_array, step_size, MPI_DOUBLE_COMPLEX, global_array,
                  sendcnts, displs, MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
    } else {

      size_t local_size = step_size / 2 / num_procs;

      Complex *local_array_even = nullptr;
      Complex *local_array_odd = nullptr;
      if (rank != num_procs-1) {
        local_array_even = new Complex[local_size];
        local_array_odd = new Complex[local_size];
      } else {
        local_array_even = new Complex[local_size + ((step_size / 2) % num_procs)]; 
        local_array_odd = new Complex[local_size + ((step_size / 2) % num_procs)]; 
      }
      for (size_t start = 0; start < size; start += step_size) {

        int sendcnts[num_procs], displs[num_procs];
        for (int i = 0; i < num_procs - 1; ++i) {
          sendcnts[i] = local_size;
          displs[i] = start + i * (local_size);
        }
        sendcnts[num_procs - 1] = (local_size) + ((step_size / 2) % num_procs);
        displs[num_procs - 1] = (num_procs - 1) * (local_size);

        MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                     local_array_even, local_size, MPI_DOUBLE_COMPLEX, root,
                     MPI_COMM_WORLD);
        MPI_Scatterv(global_array + step_size / 2, sendcnts, displs,
                     MPI_DOUBLE_COMPLEX, local_array_odd, local_size,
                     MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
        for (size_t i = 0; i < local_size; ++i) {
          Complex angle = std::pow(omega, i);
          Complex temp = local_array_even[i];
          local_array_even[i] = temp + angle * local_array_odd[i];
          local_array_odd[i] = temp - angle * local_array_odd[i];
        }

        MPI_Gatherv(local_array_even, local_size, MPI_DOUBLE_COMPLEX,
                    global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX, root,
                    MPI_COMM_WORLD);
        MPI_Gatherv(local_array_odd, local_size, MPI_DOUBLE_COMPLEX,
                    global_array + step_size / 2, sendcnts, displs,
                    MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
      }
    }
  }

  if (rank == root) {
    // bit_reverse_indices(size, 29, global_array);
    /*
    for (size_t i = 0; i < 20; ++i) {
      std::cout << global_array[i] << std::endl;
    }
    std::cout << global_array[size - 1] << std::endl;
    std::ofstream outputFile("output.dat", std::ios::binary);
    if (!outputFile.is_open()) {
      std::cerr << "Error: Failed to open file for writing.\n";
      return -1;
    }

    outputFile.write(reinterpret_cast<const char *>(global_array),
                     size * sizeof(Complex));

    outputFile.close();
    */
    delete[] global_array;
  }

  MPI_Finalize();

  return 0;
}
