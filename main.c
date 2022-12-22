#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//#include <mpi.h>

int n_samples, n_clusters, curr_nodo, n_nodos, * ind; 

float * samples_x, * samples_y, 
      * centroids_x, * centroids_y, * part_samples_x, * part_samples_y, * temp_centroids_x, * temp_centroids_y;

void alloc() {
    if(curr_nodo == 0){
        samples_x = (float *) malloc(n_samples*sizeof(float));
        samples_y = (float *) malloc(n_samples*sizeof(float));
    }

    centroids_x = (float *) malloc(n_clusters*sizeof(float));
    centroids_y = (float *) malloc(n_clusters*sizeof(float));

    part_samples_x = (float *) malloc((n_samples/n_nodos)*sizeof(float));
    part_samples_y = (float *) malloc((n_samples/n_nodos)*sizeof(float));

    temp_centroids_x = (float *) malloc(n_clusters*sizeof(float));
    temp_centroids_y = (float *) malloc(n_clusters*sizeof(float));
    
    ind = (int *) malloc(n_clusters*sizeof(int));
}

void init() {
    float x=0, y=0;

    srand(10); 

    // CREATING N SAMPLES
    for(int i = 0; i < n_samples; i++) { 
        x = (float) rand() / RAND_MAX; 
        y = (float) rand() / RAND_MAX; 

        samples_x[i] = x;
        samples_y[i] = y;
    } 

    // DIFINE FIRST CENTROID
    #pragma omp parallel for num_threads (n_threads) 
    for(int i = 0; i < n_clusters; i++) { 
        centroids_x[i] = samples_x[i]; //<<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        centroids_y[i] = samples_y[i]; //<<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  
    } 

    MPI_Bcast(&centroids_x, n_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&centroids_y, n_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatter(&samples_x, (n_samples/n_nodos), MPI_FLOAT, part_samples_x, (n_samples/n_nodos), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&samples_y, (n_samples/n_nodos), MPI_FLOAT, part_samples_y, (n_samples/n_nodos), MPI_FLOAT, 0, MPI_COMM_WORLD);
}



/*int main(int argc, char *argv[]) {
    int n_samples, n_clusters;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, n->curr_nodo); // gets this process rank
    MPI_Comm_size(MPI_COMM_WORLD, n->n_nodos); // gets number of ranks

    n_samples = atoi(argv[1]);
    n_clusters = atoi(argv[2]);
    
    // ALLOC
    alloc();

    //INIT
    if (n->curr_nodo == 0) {
        init();
        /* ... */
    } 
    
    
    MPI_Finalize();
    return 0;

}*/