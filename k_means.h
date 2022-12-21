#ifndef K_MEANS
#define K_MEANS

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include <mpi.h>

typedef struct nodo {
    float * samples_x;
    float * samples_y;
    int n_samples;

    float * centroids_x;
    float * centroids_y;
    int n_clusters;
} * Nodo;

void initNodo(Nodo n, int n_samples, int n_clusters);

#endif