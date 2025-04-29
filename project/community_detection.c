#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// Hybrid MPI+OpenMP asynchronous label propagation on an Erdős–Rényi graph,
// with adjacency‐list printing on rank 0.

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    const int    numNodes = 2304;   // total number of nodes
    const double p        = 0.0001;     // edge probabilitys

    // 1) Compute per-rank row counts & displacements
    int *countsRows = malloc(nprocs * sizeof(int));
    int *displsRows = malloc(nprocs * sizeof(int));
    int base = numNodes / nprocs, rem = numNodes % nprocs, offset = 0;
    for (int i = 0; i < nprocs; i++) {
        countsRows[i] = base + (i < rem ? 1 : 0);
        displsRows[i] = offset;
        offset += countsRows[i];
    }
    int localN = countsRows[rank];

    // 2) Rank 0 builds full adjacency, prints as adjacency list, then scatters
    int *localAdj = malloc(localN * numNodes * sizeof(int));
    if (!localAdj) { perror("malloc localAdj"); MPI_Abort(MPI_COMM_WORLD,1); }

    if (rank == 0) {
        int *fullAdj = malloc(numNodes * numNodes * sizeof(int));
        if (!fullAdj) { perror("malloc fullAdj"); MPI_Abort(MPI_COMM_WORLD,1); }
        srand(12345);
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                fullAdj[i*numNodes + j] = (rand()/(double)RAND_MAX) < p ? 1 : 0;
            }
        }

        // Print adjacency list
        // printf("Graph adjacency list:\n");
        // for (int i = 0; i < numNodes; i++) {
        //     printf("%d:", i);
        //     for (int j = 0; j < numNodes; j++) {
        //         if (fullAdj[i*numNodes + j])
        //             printf(" %d", j);
        //     }
        //     printf("\n");
        // }

        // Prepare scatterv parameters
        int *sendCounts = malloc(nprocs * sizeof(int));
        int *sendDispls = malloc(nprocs * sizeof(int));
        for (int i = 0; i < nprocs; i++) {
            sendCounts[i] = countsRows[i] * numNodes;
            sendDispls[i] = displsRows[i] * numNodes;
        }
        MPI_Scatterv(fullAdj, sendCounts, sendDispls, MPI_INT,
                     localAdj, localN*numNodes, MPI_INT,
                     0, MPI_COMM_WORLD);
        free(fullAdj);
        free(sendCounts);
        free(sendDispls);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     localAdj, localN*numNodes, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    // 3) Allocate label arrays
    int *localLabels  = malloc(localN * sizeof(int));
    int *newLocal     = malloc(localN * sizeof(int));
    int *globalLabels = malloc(numNodes * sizeof(int));
    if (!localLabels || !newLocal || !globalLabels) {
        perror("malloc labels"); MPI_Abort(MPI_COMM_WORLD,1);
    }

    // 4) Initialize global labels
    if (rank == 0) {
        for (int i = 0; i < numNodes; i++) {
            globalLabels[i] = i;
        }
    }
    MPI_Bcast(globalLabels, numNodes, MPI_INT, 0, MPI_COMM_WORLD);

    // 5) Copy to localLabels
    for (int i = 0; i < localN; i++) {
        int gidx = displsRows[rank] + i;
        localLabels[i] = globalLabels[gidx];
    }

    // 6) Asynchronous propagation until convergence
    int globalChanged;
    double t0 = MPI_Wtime();

    do {
        int localChanged = 0;

        // Update each global node in turn
        for (int gidx = 0; gidx < numNodes; gidx++) {
            int owner = 0;
            // find owner rank for gidx
            while (!(displsRows[owner] <= gidx &&
                     gidx < displsRows[owner] + countsRows[owner])) {
                owner++;
            }
            int newLabel = globalLabels[gidx];

            if (rank == owner) {
                int i = gidx - displsRows[rank];
                // count neighbor labels
                int *counts = calloc(numNodes, sizeof(int));
                int rowOff = i * numNodes;
                #pragma omp parallel for
                for (int j = 0; j < numNodes; j++) {
                    if (localAdj[rowOff + j]) {
                        #pragma omp atomic
                        counts[ globalLabels[j] ]++;
                    }
                }
                int best = localLabels[i], top = 0;
                for (int k = 0; k < numNodes; k++) {
                    if (counts[k] > top) {
                        top = counts[k];
                        best = k;
                    }
                }
                free(counts);
                if (best != localLabels[i]) {
                    localChanged = 1;
                    localLabels[i] = best;
                    newLabel = best;
                }
                globalLabels[gidx] = newLabel;
            }

            // broadcast updated label for node gidx
            MPI_Bcast(&newLabel, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (rank != owner) {
                globalLabels[gidx] = newLabel;
            }
        }

        // global convergence test
        MPI_Allreduce(&localChanged, &globalChanged, 1,
                      MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    } while (globalChanged);

    double t1 = MPI_Wtime();

    // 7) Print final labels on rank 0
    if (rank == 0) {
        printf("Final Node Labels (Community Assignments):\n");
        for (int i = 0; i < numNodes; i++) {
            printf("Node %d: Community %d\n", i, globalLabels[i]);
        }
        printf("Converged in %f seconds\n", t1 - t0);
    }

    // cleanup
    free(localAdj);
    free(localLabels);
    free(newLocal);
    free(globalLabels);
    free(countsRows);
    free(displsRows);

    MPI_Finalize();
    return 0;
}
