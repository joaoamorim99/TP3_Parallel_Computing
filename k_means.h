#ifndef K_MEANS
#define K_MEANS

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include <mpi.h>

typedef struct nodo {
    int curr_nodo;
    int n_nodos;

    float * samples_x;
    float * samples_y;
    int n_samples;

    float * centroids_x;
    float * centroids_y;
    int n_clusters;

    float * temp_centroids_x;
    float * temp_centroids_y;
    int * ind;
} * Nodo;

#endif