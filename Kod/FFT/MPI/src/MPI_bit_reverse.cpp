#include <cmath>
#include <iostream>
#include <mpi.h>
#include <mpio.h>
#include <stdlib.h>

int reverseBits(int num, int bits) {
  int reversed = 0;
  for (int i = 0; i < bits; i++) {
    reversed = (reversed << 1) | (num & 1);
    num >>= 1;
  }
  return reversed;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  int num_bits = 29;
  int size = std::pow(2, num_bits);

  MPI_Win win;
  int *global_array;
  if (rank == 0)
    MPI_Win_allocate_shared(size * sizeof(int), sizeof(int), MPI_INFO_NULL,
                            MPI_COMM_WORLD, &global_array, &win);
  if (rank != 0) {
    int disp_unit;
    MPI_Aint ssize;
    MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &global_array, &win);
    MPI_Win_shared_query(win, 0, &ssize, &disp_unit, &global_array);
  }

  if (rank == 0) {
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win);
    for (int i = 0; i < size; ++i) {
      global_array[i] = i;
    }
    MPI_Win_unlock(0, win);
  }

  int sendcnts[num_procs], displs[num_procs];

  for (int i = 0; i < num_procs - 1; ++i) {
    sendcnts[i] = size / num_procs;
    displs[i] = i * (size / num_procs);
  }
  sendcnts[num_procs - 1] = (size / num_procs) + size % num_procs;
  displs[num_procs - 1] = (num_procs - 1) * (size / num_procs);

  int *local_array = new int[sendcnts[rank]];

  MPI_Scatterv(global_array, sendcnts, displs, MPI_INT, local_array,
               sendcnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Win_lock(MPI_LOCK_SHARED, rank, MPI_MODE_NOCHECK, win);
  for (int i = 0; i < sendcnts[rank]; ++i) {
    int reversed = reverseBits(i + displs[rank], num_bits);
    global_array[reversed] = local_array[i];
  }
  MPI_Win_unlock(rank, win);

  delete[] local_array;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i=0; i<20; i++) {
      std::cout << global_array[i] << '\n';
    }

    /*
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, win);
    for (int i = 0; i < size; ++i) {
      std::cout << global_array[i] << '\n';
    }
    MPI_Win_unlock(0, win);
    */
  }

  MPI_Win_free(&win);
  MPI_Finalize();
  return 0;
}
