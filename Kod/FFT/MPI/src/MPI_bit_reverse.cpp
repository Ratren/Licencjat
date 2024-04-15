#include <cmath>
#include <stdlib.h>
#include <mpi.h>

int reverseBits(int num, int bits) {
    int reversed = 0;
    for (int i = 0; i < bits; i++) {
        reversed = (reversed << 1) | (num & 1);
        num >>= 1;
    }
    return reversed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_bits = 6;
    int num_elements = std::pow(2, 6);
    int local_elements = num_elements / size;
    int* local_data = new int[local_elements];

    free(local_data);
    MPI_Finalize();
    return 0;
}
