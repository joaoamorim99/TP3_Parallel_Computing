#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "papi.h"
#define NUM_EVENTS 7
#include <time.h>

int n_samples, n_clusters, n_threads, curr_nodo, n_nodos, * ind, * temp_ind; 

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
    #pragma omp parallel for num_threads (n_threads) 
    for(int i = 0; i < n_clusters; i++) { 
        centroids_x[i] = samples_x[i]; //<<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        centroids_y[i] = samples_y[i]; //<<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  
    } 
}

float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

void update_centroids() {
    #pragma omp parallel for num_threads (n_threads) 
    for(int i=0; i<n_clusters; i++) {

        centroids_x[i] /= ind[i];
        centroids_y[i] /= ind[i];
    }
}

void dist_all_samples(int size) {
    #pragma omp parallel for reduction (+:temp_ind[:n_clusters], temp_centroids_x[:n_clusters], temp_centroids_y[:n_clusters]) num_threads (n_threads) 
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
    n_threads = atoi(argv[3]);

    int Events[NUM_EVENTS] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM};
    long long valuesPapi[NUM_EVENTS], min_values[NUM_EVENTS];
    int retval, EventSet=PAPI_NULL;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    retval = PAPI_create_eventset(&EventSet);
    retval = PAPI_add_events(EventSet, Events, NUM_EVENTS);

    clock_t start, end;
    double cpu_time_used;
    long long start_usec, end_usec, elapsed_usec;
    start_usec = PAPI_get_real_usec();
    start = clock();


    // ALLOC
    alloc();

    //INIT
    if(curr_nodo==0) {
        generate_samples();
    }

    retval = PAPI_start(EventSet);

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

    retval = PAPI_stop(EventSet,valuesPapi);
    end_usec = PAPI_get_real_usec();
    end = clock();
    
    if(curr_nodo==0) {

        printf("N=%d, K=%d\n", n_samples, n_clusters);
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Normal: %f\n", cpu_time_used);
        elapsed_usec = end_usec - start_usec;
        printf ("\nWall clock time run: %f secs\n", (float)elapsed_usec/1000000.0);

        printf("N = %d, K = %d\n", n_samples, n_clusters);
        for(int i=0; i < n_clusters; i++) {
            printf("Center: (%.3f, %.3f) : Size %d\n", centroids_x[i], centroids_y[i], ind[i]);
        }
        printf("Iterations: %d\n", it);

        for (int i=0 ; i< NUM_EVENTS ; i++) {
            char EventCodeStr[PAPI_MAX_STR_LEN];
            if (PAPI_event_code_to_name(Events[i], EventCodeStr) == PAPI_OK) {
                fprintf (stdout, "%s = %lld\n", EventCodeStr, valuesPapi[i]);
            } else {
                fprintf (stdout, "PAPI UNKNOWN EVENT = %lld\n", valuesPapi[i]);
            }
        }
        double cpi = (double)valuesPapi[0] / (double)valuesPapi[1];
        // Print the results
        printf("CPI: %f\n", cpi);

    }
    
    MPI_Finalize();
    
    return 0;
}