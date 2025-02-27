#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* 
base case if communicator only has one process
*/
void transpose_local(double **matrix, int row_off, int col_off, int dim)
{
    for(int i =0; i<dim; i++){
        for(int j = i+1; j<dim; j++){
            double temp = matrix[row_off+i][col_off+j];
            matrix[row_off + i][col_off + j] = matrix[row_off+j][col_off+i];
            matrix[row_off +j][col_off+i] = temp;
        }
    }
}


/*
recursive transposition algorithm
*/

void recursive_transpose(double **matrix,
    int N,
    int row_off,
    int col_off,
    int dim,
    MPI_Comm comm)
{
int size, rank;
MPI_Comm_size(comm, &size);
MPI_Comm_rank(comm, &rank);

// Base case: only 1 process => local transpose
if (size == 1) {
transpose_local(matrix, row_off, col_off, dim);
return;
}

// Fallback case: if we can't split into 4 subcomms, 
// rank 0 does the entire sub-block, then broadcast.
// This prevents a divide-by-zero in new_size = size/4 when size < 4.
if (size < 4) {
if (rank == 0) {
transpose_local(matrix, row_off, col_off, dim);
}
MPI_Bcast(matrix[0], N*N, MPI_DOUBLE, 0, comm);
return;
}

// 1) Swap top-right and bottom-left blocks of current submatrix
if (rank == 0) {
int half = dim / 2;
for (int i = 0; i < half; i++) {
for (int j = 0; j < half; j++) {
int r_top    = row_off + i;
int c_right  = col_off + j + half;
int r_bottom = row_off + i + half;
int c_left   = col_off + j;

double temp = matrix[r_top][c_right];
matrix[r_top][c_right]   = matrix[r_bottom][c_left];
matrix[r_bottom][c_left] = temp;
}
}
}
MPI_Bcast(matrix[0], N*N, MPI_DOUBLE, 0, comm);

// 2) Divide the communicator into four subcommunicators
int new_size = size / 4;
int color    = rank / new_size;
MPI_Comm subcomm;
MPI_Comm_split(comm, color, rank, &subcomm);

// 3) Offsets for the sub-block handled by this subcommunicator
int half = dim / 2;
int sub_row_off = row_off + (color / 2) * half;
int sub_col_off = col_off + (color % 2) * half;

// 4) Recurse on that sub-block
recursive_transpose(matrix, N, sub_row_off, sub_col_off, half, subcomm);

MPI_Comm_free(&subcomm);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix dimension N (default = 8)
    int N = 8;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Allocate a 2D array: matrix[row][col]
    // 1) Allocate an array of row pointers
    double **matrix = (double **)malloc(N * sizeof(double *));
    // 2) Allocate the contiguous block for all rows
    matrix[0] = (double *)malloc(N * N * sizeof(double));
    // 3) Set each row pointer
    for (int i = 1; i < N; i++) {
        matrix[i] = matrix[0] + i * N;
    }

    // Fill with a simple pattern: M(i,j) = i*N + j
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (double)(i * N + j);
        }
    }

    // if (rank == 0) {
    
        
        
    //     printf("Untransposed matrix:\n");
    //     for (int i = 0; i < N; i++) {
    //         for (int j = 0; j < N; j++) {
    //             printf("%6.0f ", matrix[i][j]);
    //         }
    //         printf("\n");
    //     }
        
    // }

     // Time the actual transposition
     double start = MPI_Wtime();
     recursive_transpose(matrix, N, 0, 0, N, MPI_COMM_WORLD);
     double end = MPI_Wtime();
 
     // Print timing on rank 0
     if (rank == 0) {
         printf("Recursive Transpose completed in %f seconds\n", end - start);
 
         // Uncomment to see the transposed matrix (for small N)
         
        //  printf("Transposed matrix:\n");
        //  for (int i = 0; i < N; i++) {
        //      for (int j = 0; j < N; j++) {
        //          printf("%6.0f ", matrix[i][j]);
        //      }
        //      printf("\n");
        //  }
         
     }
 
     // Free the 2D array
     free(matrix[0]);
     free(matrix);
 
     MPI_Finalize();
     return 0;
}