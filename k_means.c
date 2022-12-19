#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int size_s, size_c, n_threads;

float *samples_x, *samples_y, *centroids_x, *centroids_y, *temp_c_x, *temp_c_y;
int *ind, *temp_ind;
 

float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

void alloc() {
    samples_x = (float *) malloc(size_s*sizeof(float));

    samples_y = (float *) malloc(size_s*sizeof(float));
    
    centroids_x = (float *) malloc(size_c*sizeof(float));

    centroids_y = (float *) malloc(size_c*sizeof(float));

    temp_c_x = (float *) malloc(size_c*sizeof(float));

    temp_c_y = (float *) malloc(size_c*sizeof(float));

    ind = (int *) malloc(size_c*sizeof(int));

    temp_ind = (int *) malloc(size_c*sizeof(int));
}

// INIT CLUSTERS 
void init() { 
    float x=0, y=0;

    srand(10); 

    // CREATING N SAMPLES
    for(int i = 0; i < size_s; i++) { 
        x = (float) rand() / RAND_MAX; 
        y = (float) rand() / RAND_MAX; 

        samples_x[i] = x;
        samples_y[i] = y;
    } 

    // DIFINE FIRST CENTROID
    #pragma omp parallel for num_threads (n_threads) 
    for(int i = 0; i < size_c; i++) { 
        centroids_x[i] = samples_x[i]; //<<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        centroids_y[i] = samples_y[i]; //<<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  

        temp_c_x[i]=0;
        temp_c_y[i]=0;

        temp_ind[i] = 0;
    } 
} 

int dist_all_samples() {
    int it;
    
    for(it=-1; it<20; it++) {

        #pragma omp parallel for reduction (+:temp_ind[:size_c], temp_c_x[:size_c], temp_c_y[:size_c]) num_threads (n_threads) 
        for(int i=0; i < size_s; i++) {
    
            int cluster = 0;
            float min = 1, dist;

            for(int j=0; j < size_c; j++) { 

                dist = euc_dist(samples_x[i], samples_y[i], centroids_x[j], centroids_y[j]);
                
                if(dist<min) {
                    min = dist;
                    cluster = j;
                }
            }

            temp_c_x[cluster] += samples_x[i];
            temp_c_y[cluster] += samples_y[i];

            temp_ind[cluster]++;
        }

        #pragma omp parallel for num_threads (n_threads) 
        for(int i=0; i<size_c; i++) {

            centroids_x[i] = (temp_c_x[i] / temp_ind[i]);
            centroids_y[i] = (temp_c_y[i] / temp_ind[i]);

            temp_c_x[i] = 0;
            temp_c_y[i] = 0;

            ind[i] = temp_ind[i];
            temp_ind[i] = 0;
        }
    }
    return it;
}

int main(int argc, char *argv[]) {
    int rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);



    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets this process rank
    
    /* Process 0 sends and Process 1 receives */
    if (rank == 0) {
        msg = 123456;
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } // (buf, count, datatype, dest, tag, comm)
    
    else if (rank == 1) {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        printf("Received %d\n", msg);  
    }
    
    MPI_Finalize();
    return 0;

}