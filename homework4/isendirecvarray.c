/****************************************************************
 ****
 **** This program file is part of the book 
 **** `Parallel programming for Science and Engineering'
 **** by Victor Eijkhout, eijkhout@tacc.utexas.edu
 ****
 **** copyright Victor Eijkhout 2012-2021
 ****
 **** MPI Exercise for Isend/Irecv, sending an array
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#include "tools.h"

int main(int argc,char **argv) {

  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int nprocs, procno;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procno);

#define N 100
  double indata[N], outdata[N];
  for (int i=0; i<N; i++)
    indata[i] = 1.;

  double leftdata = 0., rightdata = 0.;
  MPI_Request requests[4];

  /* 
   * We want the following ghost exchange:
   * - For a process p with p>0, its left ghost (to be used at index 0)
   *   should be the rightmost element of process p-1 (i.e. indata[N-1] of p-1).
   * - For a process p with p<nprocs-1, its right ghost (to be used at index N-1)
   *   should be the leftmost element of process p+1 (i.e. indata[0] of p+1).
   *
   * To do so, we arrange that:
   *   • In the “left neighbor” block, every process sends its leftmost element
   *     to its left neighbor. (That message will eventually be used as the right ghost
   *     of that neighbor.)
   *   • In the “right neighbor” block, every process sends its rightmost element
   *     to its right neighbor. (That message will eventually be used as the left ghost
   *     of that neighbor.)
   *
   * But each process also posts receives:
   *   • In the left block, process p receives (into leftdata) from its left neighbor.
   *     In fact, process p–1 (if it exists) will send its rightmost element in its right–block.
   *   • In the right block, process p receives (into rightdata) from its right neighbor.
   *     In fact, process p+1 (if it exists) will send its leftmost element in its left–block.
   */

  /* ----- left neighbor exchange ----- */
  int left_sendto = (procno == 0) ? MPI_PROC_NULL : procno - 1;
  int left_recvfrom = (procno == 0) ? MPI_PROC_NULL : procno - 1;
  /* Send my leftmost element so that my left neighbor may use it as its right ghost */
  MPI_Isend(&indata[0], 1, MPI_DOUBLE, left_sendto, 0, comm, &requests[0]);
  /* Receive left ghost (which is actually the rightmost element of my left neighbor) */
  MPI_Irecv(&leftdata, 1, MPI_DOUBLE, left_recvfrom, 0, comm, &requests[1]);

  /* ----- right neighbor exchange ----- */
  int right_sendto = (procno == nprocs - 1) ? MPI_PROC_NULL : procno + 1;
  int right_recvfrom = (procno == nprocs - 1) ? MPI_PROC_NULL : procno + 1;
  /* Send my rightmost element so that my right neighbor may use it as its left ghost */
  MPI_Isend(&indata[N-1], 1, MPI_DOUBLE, right_sendto, 0, comm, &requests[2]);
  /* Receive right ghost (which is actually the leftmost element of my right neighbor) */
  MPI_Irecv(&rightdata, 1, MPI_DOUBLE, right_recvfrom, 0, comm, &requests[3]);

  /* Wait for all nonblocking communications to complete */
  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

  /*
   * Do the averaging operation.
   * Note that leftdata==rightdata==0 if not explicitly received.
   */
  for (int i=0; i<N; i++)
    if (i==0)
      outdata[i] = leftdata + indata[i] + indata[i+1];
    else if (i==N-1)
      outdata[i] = indata[i-1] + indata[i] + rightdata;
    else
      outdata[i] = indata[i-1] + indata[i] + indata[i+1];
  
  /*
   * Check correctness of the result:
   * value should be 2 at the end points, 3 everywhere else.
   */
  double answer[N];
  for (int i=0; i<N; i++) {
    if ( (procno==0 && i==0) || (procno==nprocs-1 && i==N-1) ) {
      answer[i] = 2.;
    } else {
      answer[i] = 3.;
    }
  }
  int error_test = array_error(answer,outdata,N);
  print_final_result(error_test,comm);

  MPI_Finalize();
  return 0;
}
