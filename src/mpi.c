#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int n_samples, n_clusters, curr_nodo, n_nodos, * ind, * temp_ind; 

float * samples_x, * samples_y, 
      * centroids_x, * centroids_y, * part_samples_x, * part_samples_y, * temp_centroids_x, * temp_centroids_y;

void alloc() {

    samples_x = (float *) malloc(n_samples*sizeof(float));
    samples_y = (float *) malloc(n_samples*sizeof(float));

    centroids_x = (float *) malloc(n_clusters*sizeof(float));
    centroids_y = (float *) malloc(n_clusters*sizeof(float));

    part_samples_x = (float *) malloc((n_samples/n_nodos)*sizeof(float));
    part_samples_y = (float *) malloc((n_samples/n_nodos)*sizeof(float));

    temp_centroids_x = (float *) calloc(n_clusters, sizeof(float));
    temp_centroids_y = (float *) calloc(n_clusters, sizeof(float));
    
    ind = (int *) calloc(n_clusters, sizeof(int));
    temp_ind = (int *) calloc(n_clusters, sizeof(int));
}

void generate_samples() {
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
    for(int i = 0; i < n_clusters; i++) { 
        centroids_x[i] = samples_x[i]; //<<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        centroids_y[i] = samples_y[i]; //<<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  
    } 
}

float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

void update_centroids() {
    for(int i=0; i<n_clusters; i++) {

        centroids_x[i] /= ind[i];
        centroids_y[i] /= ind[i];
    }
}

int dist_all_samples(int size) {
    for(int i=0; i < size; i++) {

        int cluster = 0;
        float min = 1, dist;

        for(int j=0; j < n_clusters; j++) { 

            dist = euc_dist(part_samples_x[i], part_samples_y[i], centroids_x[j], centroids_y[j]);
            
            if(dist<min) {
                min = dist;
                cluster = j;
            }
        }

        temp_centroids_x[cluster] += part_samples_x[i];
        temp_centroids_y[cluster] += part_samples_y[i];

        temp_ind[cluster]++;
    }
}

//srun --partition=cpar --ntasks=4 perf stat mpirun -np 4 ./k_means 10000000 4

int main(int argc, char *argv[]) {
    int it;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_nodo); // gets this process rank
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodos); // gets number of ranks

    n_samples = atoi(argv[1]);
    n_clusters = atoi(argv[2]);

    // ALLOC
    alloc();

    //INIT
    if(curr_nodo==0) {
        generate_samples();
    }

    MPI_Scatter(samples_x, (n_samples/n_nodos), MPI_FLOAT, part_samples_x, (n_samples/n_nodos), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(samples_y, (n_samples/n_nodos), MPI_FLOAT, part_samples_y, (n_samples/n_nodos), MPI_FLOAT, 0, MPI_COMM_WORLD);

    for(it=-1; it<20; it++) {
        MPI_Bcast(centroids_x, n_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroids_y, n_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD);

        dist_all_samples((n_samples/n_nodos));

        MPI_Reduce(temp_centroids_x, centroids_x, n_clusters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(temp_centroids_y, centroids_y, n_clusters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(temp_ind, ind, n_clusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(curr_nodo==0) update_centroids();

        for(int i=0; i<n_clusters; i++) {
            temp_centroids_x[i] = 0;
            temp_centroids_y[i] = 0;

            temp_ind[i] = 0;
        }
    }
    
    if(curr_nodo==0) {
        printf("N = %d, K = %d\n", n_samples, n_clusters);
        for(int i=0; i < n_clusters; i++) {
            printf("Center: (%.3f, %.3f) : Size %d\n", centroids_x[i], centroids_y[i], ind[i]);
        }
        printf("Iterations: %d\n", it);
    }
    
    MPI_Finalize();
    
    return 0;
}