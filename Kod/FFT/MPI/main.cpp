#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

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

void random_values_into_array(Complex *array, int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (int i = 0; i < size; ++i) {
    array[i] = Complex(dis(gen), 0);
  }
}

void init_random_values(Complex *global_array, int size, int rank, int root,
                        int num_procs) {

  Complex *local_array = nullptr;

  int local_size;

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

void bit_reverse_indices(MPI_Win &win, Complex *global_array, int size,
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

  int num_bits;
  int size;
  MPI_Win win;
  Complex *global_array;

  if (rank == 0) {
    std::ifstream inputFile("./TEST_DATA/FFT_test_vector",
                            std::ios::binary);
    if (!inputFile.is_open()) {
      std::cerr << "Error: Failed to open file for reading.\n";
      return -1;
    }

    inputFile.seekg(0, std::ios::end);
    long fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    size = fileSize / sizeof(Complex);
    
  }

  MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
  num_bits = log2(size);

  if (rank == 0) {
    std::ifstream inputFile("./TEST_DATA/FFT_test_vector",
                            std::ios::binary);
    if (!inputFile.is_open()) {
      std::cerr << "Error: Failed to open file for reading.\n";
      return -1;
    }

    inputFile.seekg(0, std::ios::end);
    long fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    MPI_Win_allocate_shared(size * sizeof(Complex), sizeof(Complex),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &global_array, &win);

    inputFile.read(reinterpret_cast<char *>(global_array), fileSize);

    inputFile.close();

  } else {
    int disp_unit;
    MPI_Aint ssize;
    MPI_Win_allocate_shared(0, sizeof(Complex), MPI_INFO_NULL, MPI_COMM_WORLD,
                            &global_array, &win);
    MPI_Win_shared_query(win, 0, &ssize, &disp_unit, &global_array);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  bit_reverse_indices(win, global_array, size, rank, root, num_procs, num_bits);

  int *sendcnts = new int[num_procs]();
  int *displs = new int[num_procs]();
  int *displs_odd = new int[num_procs]();
  for (int i = 0; i < num_procs; ++i) {
    sendcnts[i] = 0;
    displs[i] = 0;
  }
  int closest_pow_2_num_procs = 1;
  while (closest_pow_2_num_procs < num_procs) {
    closest_pow_2_num_procs <<= 1;
  }
  int base_num = size / closest_pow_2_num_procs;
  int additional =
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

  std::vector<Complex> local_array(sendcnts[rank]);
  std::vector<Complex> local_array_odd;

  double start_time = MPI_Wtime();

  MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX,
               local_array.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX, root,
               MPI_COMM_WORLD);

  Complex omega, omega_power, temp;
  int step_size, index_even, index_odd;
  for (int i = 1; i <= num_bits; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    step_size = 1 << i;
    omega = std::exp(-2.0 * J * M_PI / (double)step_size);
    if (i < num_bits - 5) {
      // DOPÓKI operacja sendcnts[rank] % step_size == 0 nie jest prawdziwa,
      // czyli kroki mieszczą się dobrze w podzielonym miejscu
      for (int start = 0; start < sendcnts[rank]; start += step_size) {
        Complex omega_power = 1;
        for (int step = 0; step < step_size / 2; ++step) {
          index_even = start + step;
          index_odd = start + step + step_size / 2;
          temp = local_array[index_even];
          local_array[index_even] += omega_power * local_array[index_odd];
          local_array[index_odd] = temp - omega_power * local_array[index_odd];
          omega_power *= omega;
        }
      }
    } else {
      if (i == num_bits - 5) {
        // ZGROMADŹ DANE PRZY PIERWSZYM KROKU, KTÓRY
        // NIE MIESCI SIĘ W PODZIELONYM MIEJSCU
        MPI_Gatherv(local_array.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX,
                    global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX, root,
                    MPI_COMM_WORLD);
        local_array.resize((size/num_procs/2) + num_procs);
        local_array_odd.resize((size/num_procs/2) + num_procs);
      }
      // JESLI KROKI NIE MIESZCZĄ SIE W PODZIELONYM MIEJSCU TO:

      for (int start = 0; start < size; start += step_size) {
        // Dla każdego kroku podział danych.
        base_num = step_size / 2 / num_procs;
        additional = (step_size / 2) - base_num * num_procs;
        for (int j = 0; j < num_procs - 1; ++j) {
          sendcnts[j] = base_num;
          displs[j] = j * base_num + start;
          displs_odd[j] = j * base_num + start + (step_size / 2);
        }
        sendcnts[num_procs - 1] = base_num + additional;
        displs[num_procs - 1] = (num_procs - 2) * base_num + start;
        displs_odd[num_procs - 1] =
            (num_procs - 2) * base_num + start + (step_size / 2);

        MPI_Scatterv(global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX,
                     local_array.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX,
                     root, MPI_COMM_WORLD);
        MPI_Scatterv(global_array, sendcnts, displs_odd, MPI_DOUBLE_COMPLEX,
                     local_array_odd.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX,
                     root, MPI_COMM_WORLD);
        omega_power = std::pow(omega, base_num * rank);
        for (int step = 0; step < sendcnts[rank]; ++step) {
          temp = local_array[step];
          local_array[step] += omega_power * local_array_odd[step];
          local_array_odd[step] = temp - omega_power * local_array_odd[step];
          omega_power *= omega;
        }
        MPI_Gatherv(local_array.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX,
                    global_array, sendcnts, displs, MPI_DOUBLE_COMPLEX, root,
                    MPI_COMM_WORLD);
        MPI_Gatherv(local_array_odd.data(), sendcnts[rank], MPI_DOUBLE_COMPLEX,
                    global_array, sendcnts, displs_odd, MPI_DOUBLE_COMPLEX, root,
                    MPI_COMM_WORLD);
      }
    }
  }

  double end_time = MPI_Wtime();

  if (rank == 0) {
    std::cout << "FFT zostało wykonane w czasie: " << end_time - start_time << " sekund\n";
  }

  delete[] displs;
  delete[] sendcnts;
  delete[] displs_odd;

  MPI_Win_free(&win);
  MPI_Finalize();

  return 0;
}
