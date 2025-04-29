#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

// Graph structure to hold adjacency matrix and labels
typedef struct {
    int *labels;       // Labels for each node in the graph
    int numVertices;   // Number of nodes in the graph
    int **adjMatrix;   // Adjacency matrix representing the graph
} Graph;

// Function to generate a random graph using the Erdős–Rényi model
Graph* generateGraph(int n, double p) {
    Graph *graph = malloc(sizeof(Graph));
    graph->numVertices = n;
    graph->adjMatrix = malloc(n * sizeof(int *));
    graph->labels = malloc(n * sizeof(int));  // Labels for community detection

    for (int i = 0; i < n; i++) {
        graph->adjMatrix[i] = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            graph->adjMatrix[i][j] = (rand() / (double)RAND_MAX) < p ? 1 : 0;
        }
    }
    return graph;
}

// Function to partition the graph for each MPI process
Graph* partitionGraph(Graph *graph, int numProcesses, int processRank) {
    int nodesPerProcess = graph->numVertices / numProcesses;
    Graph *subgraph = malloc(sizeof(Graph));
    subgraph->numVertices = nodesPerProcess;
    subgraph->adjMatrix = malloc(nodesPerProcess * sizeof(int *));
    subgraph->labels = malloc(nodesPerProcess * sizeof(int));

    for (int i = processRank * nodesPerProcess; i < (processRank + 1) * nodesPerProcess; i++) {
        subgraph->adjMatrix[i - processRank * nodesPerProcess] = graph->adjMatrix[i];
        subgraph->labels[i - processRank * nodesPerProcess] = graph->labels[i];
    }

    return subgraph;
}

// Boundary communication between MPI processes
void boundaryCommunication(int *boundaryLabels, int boundaryCount, int rank, int numProcesses) {
    MPI_Request sendRequest, recvRequest;

    for (int i = 0; i < numProcesses; i++) {
        if (i != rank) {
            MPI_Isend(boundaryLabels, boundaryCount, MPI_INT, i, 0, MPI_COMM_WORLD, &sendRequest);
            MPI_Irecv(boundaryLabels, boundaryCount, MPI_INT, i, 0, MPI_COMM_WORLD, &recvRequest);
        }
    }

    MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
    MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);
}

// Label propagation algorithm
void labelPropagation(Graph *subgraph, int maxIterations, double convergenceThreshold, int processRank, int numProcesses) {
    int *newLabels = malloc(subgraph->numVertices * sizeof(int));
    int converged = 0;

    // Initialize labels (each node starts with a unique label)
    for (int i = 0; i < subgraph->numVertices; i++) {
        subgraph->labels[i] = i;
    }

    int iteration = 0;
    while (iteration < maxIterations && !converged) {
        converged = 1;

        // Label propagation step (parallel within process using OpenMP)
        #pragma omp parallel for shared(subgraph)
        for (int i = 0; i < subgraph->numVertices; i++) {
            // Count the frequency of labels among neighbors
            int *neighborLabels = (int*) calloc(subgraph->numVertices, sizeof(int));

            // Check the neighbors of node i (based on the adjacency matrix)
            for (int j = 0; j < subgraph->numVertices; j++) {
                if (subgraph->adjMatrix[i][j] == 1) {  // If there's an edge between node i and node j
                    neighborLabels[subgraph->labels[j]]++;  // Increment the count for the label of node j
                }
            }

            // Find the most frequent label among the neighbors
            int maxCount = 0;
            int mostFrequentLabel = subgraph->labels[i];  // Default to the current label
            for (int j = 0; j < subgraph->numVertices; j++) {
                if (neighborLabels[j] > maxCount) {
                    maxCount = neighborLabels[j];
                    mostFrequentLabel = j;
                }
            }

            // Update the label of node i
            newLabels[i] = mostFrequentLabel;

            // Free the memory for neighbor labels
            free(neighborLabels);
        }

        // Check for convergence (if labels haven't changed)
        for (int i = 0; i < subgraph->numVertices; i++) {
            if (subgraph->labels[i] != newLabels[i]) {
                converged = 0;  // Labels have changed, so not converged yet
                break;
            }
        }

        // If not converged, copy the new labels back to subgraph labels
        for (int i = 0; i < subgraph->numVertices; i++) {
            subgraph->labels[i] = newLabels[i];
        }

        iteration++;
    }

    free(newLabels);
}

// Function to evaluate performance and measure execution time
void evaluatePerformance(int numProcesses) {
    double startTime = MPI_Wtime();  // Start time

    // Here you could run the graph generation, partitioning, and label propagation

    double endTime = MPI_Wtime();  // End time
    if (numProcesses == 1) {  // Print only from rank 0
        printf("Total Execution Time: %f seconds\n", endTime - startTime);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numProcesses, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);  // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);  // Get the rank of this process

    // Record the start time for the program
    double startTime = MPI_Wtime();  // Start time for execution

    // Graph generation
    int numNodes = 10000;
    double edgeProbability = 0.1;
    Graph* graph = generateGraph(numNodes, edgeProbability);

    // Partition the graph for each process
    Graph* subgraph = partitionGraph(graph, numProcesses, processRank);

    // Run label propagation algorithm
    int maxIterations = 100;
    double convergenceThreshold = 0.01;
    labelPropagation(subgraph, maxIterations, convergenceThreshold, processRank, numProcesses);

    // Record the end time for the program
    double endTime = MPI_Wtime();  // End time for execution

    // Print total execution time from rank 0 only
    if (processRank == 0) {
        printf("Total Execution Time: %f seconds\n", endTime - startTime);
    }

    // Performance evaluation (optional)
    evaluatePerformance(numProcesses);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
