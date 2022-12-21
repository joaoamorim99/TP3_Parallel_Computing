#include "k_means.h"

float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

void initNodo(Nodo n, int n_samples, int n_clusters) {
    n->samples_x = (float *) malloc(n_samples*sizeof(float));
    n->samples_y = (float *) malloc(n_samples*sizeof(float));
    n->n_samples = n_samples;

    n->centroids_x = (float *) malloc(n_clusters*sizeof(float));
    n->centroids_y = (float *) malloc(n_clusters*sizeof(float));
    n->n_clusters = n_clusters;

    float x=0, y=0;

    srand(10); 

    // CREATING N SAMPLES
    for(int i = 0; i < n_samples; i++) { 
        x = (float) rand() / RAND_MAX; 
        y = (float) rand() / RAND_MAX; 

        n->samples_x[i] = x;
        n->samples_y[i] = y;
    } 

    // DIFINE FIRST CENTROID
    #pragma omp parallel for num_threads (n_threads) 
    for(int i = 0; i < n_clusters; i++) { 
        n->centroids_x[i] = n->samples_x[i]; //<<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        n->centroids_y[i] = n->samples_y[i]; //<<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  
        //MPI_SEND or BROADCAST
    } 
}

/*int dist_all_samples() {
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
}*/