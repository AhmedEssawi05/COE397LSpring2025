#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* 
Base case: If communicator only has one process, perform a local in-place transpose.
*/
void transpose_local(double **matrix, int row_off, int col_off, int dim)
{
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            double temp = matrix[row_off + i][col_off + j];
            matrix[row_off + i][col_off + j] = matrix[row_off + j][col_off + i];
            matrix[row_off + j][col_off + i] = temp;
        }
    }
}

/*
Recursive transposition algorithm with parallelized block swapping.
*/
void recursive_transpose(double **matrix, int N, int row_off, int col_off, int dim, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // Base case: Only one process in this communicator.
    if (size == 1) {
        transpose_local(matrix, row_off, col_off, dim);
        return;
    }

    // Fallback case: If we can't split into 4 subcommunicators, do the local transpose on rank 0 and broadcast.
    if (size < 4) {
        if (rank == 0) {
            transpose_local(matrix, row_off, col_off, dim);
        }
        MPI_Bcast(matrix[0], N * N, MPI_DOUBLE, 0, comm);
        return;
    }

    // Parallelized swap of top-right and bottom-left blocks.
    int half = dim / 2;
    int total_swaps = half * half;  // Total number of swap operations.
    int swaps_per_proc = (total_swaps + size - 1) / size;  // Ceiling division to distribute swaps evenly.
    int start_swap = rank * swaps_per_proc;
    int end_swap = start_swap + swaps_per_proc;
    if (end_swap > total_swaps)
        end_swap = total_swaps;
    
    // Each process performs its assigned swap operations.
    for (int idx = start_swap; idx < end_swap; idx++) {
        int i = idx / half;
        int j = idx % half;
        int r_top    = row_off + i;
        int c_right  = col_off + j + half;
        int r_bottom = row_off + i + half;
        int c_left   = col_off + j;
        double temp = matrix[r_top][c_right];
        matrix[r_top][c_right] = matrix[r_bottom][c_left];
        matrix[r_bottom][c_left] = temp;
    }
    // Synchronize all processes before moving forward.
    MPI_Barrier(comm);

    // Divide the communicator into four subcommunicators.
    int new_size = size / 4;
    int color = rank / new_size;
    MPI_Comm subcomm;
    MPI_Comm_split(comm, color, rank, &subcomm);

    // Determine the offsets for the submatrix handled by this subcommunicator.
    int sub_row_off = row_off + (color / 2) * half;
    int sub_col_off = col_off + (color % 2) * half;

    // Recursively transpose the designated submatrix.
    recursive_transpose(matrix, N, sub_row_off, sub_col_off, half, subcomm);

    MPI_Comm_free(&subcomm);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix dimension N (default = 8, or specified via command-line argument).
    int N = 8;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Allocate a 2D array: matrix[row][col]
    double **matrix = (double **)malloc(N * sizeof(double *));
    matrix[0] = (double *)malloc(N * N * sizeof(double));
    for (int i = 1; i < N; i++) {
        matrix[i] = matrix[0] + i * N;
    }

    // Initialize matrix with a simple pattern: M(i,j) = i * N + j.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (double)(i * N + j);
        }
    }

    // Time the transposition.
    double start = MPI_Wtime();
    recursive_transpose(matrix, N, 0, 0, N, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // Print timing on rank 0.
    if (rank == 0) {
        printf("Recursive Transpose completed in %f seconds\n", end - start);
        // Uncomment below to print the transposed matrix (suitable for small N).
        
        // printf("Transposed matrix:\n");
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < N; j++) {
        //         printf("%6.0f ", matrix[i][j]);
        //     }
        //     printf("\n");
        // }
        
    }

    free(matrix[0]);
    free(matrix);
    MPI_Finalize();
    return 0;
}
