#include <complex>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>

#define J Complex(0.0, 1.0)

typedef std::complex<double> Complex;

int reverse_bits(int num, int bits) {
  int reversed = 0;
  for (int i = 0; i < bits; i++) {
    reversed = (reversed << 1) | (num & 1);
    num >>= 1;
  }
  return reversed;
}

void random_values_into_array(Complex *array, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (size_t i = 0; i < size; ++i) {
    array[i] = Complex(dis(gen), 0);
  }
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

void bit_reverse_indices(MPI_Win &win, Complex *global_array, size_t size,
                         int rank, int root, int num_procs, int num_bits) {
  int *sendcnts = new int[num_procs];
  int *displs = new int[num_procs];

  for (int i = 0; i < num_procs - 1; ++i) {
    sendcnts[i] = size / num_procs;
    displs[i] = i * (size / num_procs);
  }
  sendcnts[num_procs - 1] = (size / num_procs) + size % num_procs;
  displs[num_procs - 1] = (num_procs - 1) * (size / num_procs);

  Complex *local_array = new Complex[sendcnts[rank]];

  MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX, local_array,
               sendcnts[rank], MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, win);
  for (int i = 0; i < sendcnts[rank]; ++i) {
    global_array[reverse_bits(i + displs[rank], num_bits)] = local_array[i];
  }
  MPI_Win_unlock(rank, win);

  delete[] local_array;
  delete[] sendcnts;
  delete[] displs;

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, num_procs, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  size_t num_bits = 29;
  size_t size = std::pow(2, num_bits);
  MPI_Win win;
  Complex *global_array;

  if (rank == 0) {
    MPI_Win_allocate_shared(size * sizeof(Complex), sizeof(Complex),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &global_array, &win);
  } else {
    int disp_unit;
    MPI_Aint ssize;
    MPI_Win_allocate_shared(0, sizeof(Complex), MPI_INFO_NULL, MPI_COMM_WORLD,
                            &global_array, &win);
    MPI_Win_shared_query(win, 0, &ssize, &disp_unit, &global_array);
  }

  if (rank == 0) {
    std::ifstream inputFile("input.dat", std::ios::binary);
    if (!inputFile.is_open()) {
      std::cerr << "Error: Failed to open file for reading.\n";
      return -1;
    }

    inputFile.seekg(0, std::ios::end);
    std::streampos fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    size = fileSize / sizeof(Complex);

    inputFile.read(reinterpret_cast<char *>(global_array), fileSize);

    inputFile.close();

    for (int i=0; i<20; ++i) {
      std::cout << global_array[i] << '\t';
    }
  }

  // init_random_values(global_array, size, rank, root, num_procs);
  MPI_Barrier(MPI_COMM_WORLD);
  bit_reverse_indices(win, global_array, size, rank, root, num_procs, num_bits);

  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t step = 1; step <= num_bits; ++step) {
    size_t step_size = 1 << step;
    Complex omega = std::exp((2 * M_PI * J) / static_cast<double>(step_size));
    size_t num_chunks = size / step_size;

    if (rank == 0) {
      for (int i=0; i<20; ++i) {
        std::cout << global_array[i] << '\t';
      }
      std::cout << "\n\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (num_chunks >= 16) {
      int sendcnts[num_procs], displs[num_procs];
      for (int i = 0; i < num_procs; ++i) {
        sendcnts[i] = 0;
        displs[i] = 0;
      }
      int closest_pow_2_num_procs = 1;
      while (closest_pow_2_num_procs < num_procs) {
        closest_pow_2_num_procs <<= 1;
      }
      size_t base_num = size / closest_pow_2_num_procs;
      size_t additional =
          (size - base_num * num_procs) / (closest_pow_2_num_procs >> 1);

      for (int i = 0; i < num_procs; ++i) {
        sendcnts[i] = base_num;
        if (i < (closest_pow_2_num_procs >> 1)) {
          sendcnts[i] += additional;
        }
        for (int j = i + 1; j < num_procs; ++j) {
          displs[j] += sendcnts[i];
        }
      }

      Complex *local_array = new Complex[sendcnts[rank]];

      MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                   local_array, sendcnts[rank], MPI_DOUBLE_COMPLEX, root,
                   MPI_COMM_WORLD);

      for (size_t start = 0; start < sendcnts[rank]; start += step_size) {
        for (size_t i = 0; i < step_size / 2; i++) {
          size_t index_even = start + i;
          size_t index_odd = index_even + step_size / 2;
          Complex temp = local_array[index_even];
          Complex angle = std::pow(omega, i);
          local_array[index_even] = temp + angle * local_array[index_odd];
          local_array[index_odd] = temp - angle * local_array[index_odd];
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);

      MPI_Gatherv(local_array, sendcnts[rank], MPI_DOUBLE_COMPLEX, global_array,
                  sendcnts, displs, MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);

      delete[] local_array;

    } else {

      int *sendcnts = new int[num_procs];
      int *displs = new int[num_procs];

      for (int i = 0; i < num_procs - 1; ++i) {
        sendcnts[i] = (step_size / 2) / num_procs;
        displs[i] = i * ((step_size / 2) / num_procs);
      }
      sendcnts[num_procs - 1] =
          ((step_size / 2) / num_procs) + (step_size / 2) % num_procs;
      displs[num_procs - 1] = (num_procs - 1) * ((step_size / 2) / num_procs);

      Complex *even_array = new Complex[sendcnts[rank]];
      Complex *odd_array = new Complex[sendcnts[rank]];
      for (size_t start = 0; start < size; start += step_size) {

        MPI_Scatterv(global_array + start, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                     even_array, sendcnts[rank], MPI_DOUBLE_COMPLEX, root,
                     MPI_COMM_WORLD);

        MPI_Scatterv(global_array + start + (step_size / 2), sendcnts, displs,
                     MPI_DOUBLE_COMPLEX, odd_array, sendcnts[rank],
                     MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);

        for (size_t i = 0; i < sendcnts[rank]; ++i) {
          Complex temp = even_array[i];
          Complex angle = std::pow(omega, i+displs[rank]);
          even_array[i] = temp + angle * odd_array[i];
          odd_array[i] = temp - angle * odd_array[i];
        }

        MPI_Gatherv(even_array, sendcnts[rank], MPI_DOUBLE_COMPLEX,
                    global_array + start, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                    root, MPI_COMM_WORLD);

        MPI_Gatherv(odd_array, sendcnts[rank], MPI_DOUBLE_COMPLEX,
                    global_array + start + (step_size / 2), sendcnts, displs,
                    MPI_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
      }

      delete[] even_array;
      delete[] odd_array;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::ofstream outputFile("output.dat", std::ios::binary);
    if (!outputFile.is_open()) {
      std::cerr << "Error: Failed to open file for writing.\n";
      return -1;
    }

    outputFile.write(reinterpret_cast<const char *>(global_array),
                     size * sizeof(Complex));

    outputFile.close();
  }

  MPI_Win_free(&win);
  MPI_Finalize();

  return 0;
}
