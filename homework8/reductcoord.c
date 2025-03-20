/******************************************************************************
 * reductcoord.c
 *
 * Exercise 8 from the OpenMP course.
 *
 * This program finds the coordinate (x, y) with the maximal Euclidean norm,
 * i.e. with the maximum value of sqrt(x^2 + y^2), among an array of randomly
 * generated coordinates. We use a user-defined reduction to combine candidates
 * from different threads.
 *
 * We store the coordinates in an array of struct coord.
 *
 * Compilation:
 *     gcc -fopenmp reductcoord.c -o reductcoord -lm
 *
 * Usage:
 *     ./reductcoord [N]
 * where [N] is the optional number of coordinates (default is 1,000,000).
 ******************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <omp.h>
 #include <time.h>
 
 /* Define a structure to hold a coordinate (x, y) */
 typedef struct {
     double x;
     double y;
 } coord;
 
 /*
  * Declare a user-defined reduction operator "maxCoord" for type coord.
  *
  * The combiner compares the squared norms of two coordinates
  * (i.e. (x*x + y*y)) and assigns to omp_out the coordinate with the larger norm.
  *
  * The initializer sets each thread’s private copy (omp_priv) to {0.0, 0.0}.
  *
  * Note: This syntax and idea is taken directly from the slides on user-defined
  * reductions (see slide 60 and Exercise 8).
  */
 #pragma omp declare reduction(maxCoord : coord : \
     omp_out = ((omp_in.x*omp_in.x + omp_in.y*omp_in.y) > (omp_out.x*omp_out.x + omp_out.y*omp_out.y) ? (omp_out = omp_in) : omp_out) ) \
     initializer(omp_priv = {0.0, 0.0})
 
 int main(int argc, char *argv[]) {
     int N = 1000000;  // default number of coordinates
     if (argc > 1) {
         N = atoi(argv[1]);
     }
 
     /* Allocate an array of N coordinates (stored as struct coord) */
     coord *coords = (coord *) malloc(N * sizeof(coord));
     if (coords == NULL) {
         fprintf(stderr, "Error: Could not allocate memory for coordinates.\n");
         return 1;
     }
 
     /* Seed the random number generator */
     srand(time(NULL));
 
     /* Generate N random coordinates in the range [0, 1] */
     for (int i = 0; i < N; i++) {
         coords[i].x = (double)rand() / RAND_MAX;
         coords[i].y = (double)rand() / RAND_MAX;
     }
 
     /* This variable will hold the coordinate with the maximum norm.
      * Its initial value is {0.0, 0.0} which is the identity for our reduction.
      */
     coord max_coord = {0.0, 0.0};
 
     /*
      * Parallelize the loop using OpenMP with our custom reduction.
      * Each iteration "submits" a candidate coordinate (coords[i]) to the reduction.
      * The reduction operator defined above compares the squared norms and
      * retains the coordinate with the larger value.
      */
     #pragma omp parallel for reduction(maxCoord:max_coord)
     for (int i = 0; i < N; i++) {
         /* The assignment here simply provides a candidate for the reduction. */
         max_coord = coords[i];
     }
 
     /* Compute the Euclidean norm of the maximum coordinate */
     double max_norm = sqrt(max_coord.x * max_coord.x + max_coord.y * max_coord.y);
 
     /* Print the result */
     printf("Coordinate with max norm: (%.6f, %.6f)\n", max_coord.x, max_coord.y);
     printf("Max norm = %.6f\n", max_norm);
 
     /* Clean up allocated memory */
     free(coords);
     return 0;
 }
 