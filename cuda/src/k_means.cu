#include "k_means.h"

#define NUM_BLOCKS 1020
#define NUM_THREADS_PER_BLOCK 1024
#define N NUM_BLOCKS*NUM_THREADS_PER_BLOCK 
using namespace std;

// HOST
float * samples_x, * samples_y, * centroids_x, * centroids_y, * temp_centroids_x, * temp_centroids_y;
int n_samples, n_clusters, * ind, * temp_ind;;

__device__ float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

// KERNEL
__global__ void k_means(int *mn_samples, int * mn_clusters, 
							float * msamples_x, float * msamples_y, 
							float * mcentroids_x, float * mcentroids_y, 
							float * mtemp_centroids_x,  float * mtemp_centroids_y, 
							int * mtemp_ind, int * mind) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= (*mn_samples)) return; 
	
	int cluster = 0;
	float min=1, dist;

	for(int j=0; j < (*mn_clusters); j++) {
		dist = euc_dist(msamples_x[id], msamples_y[id], mcentroids_x[j], mcentroids_y[j]);
			
		if(dist<min) {
			min = dist;
			cluster = j;
		}
	}
	
	atomicAdd(&mtemp_centroids_x[cluster], msamples_x[id]);
	atomicAdd(&mtemp_centroids_y[cluster], msamples_y[id]);
	atomicAdd(&mtemp_ind[cluster], 1);
}

__global__ void updateCentroids(int *mn_clusters,
								 float * mcentroids_x, float * mcentroids_y, 
								 float * mtemp_centroids_x,  float * mtemp_centroids_y, 
								 int * mtemp_ind, int * mind) {

	for(int i=0; i < (*mn_clusters); i++) {
		
		mcentroids_x[i] = (mtemp_centroids_x[i] / mtemp_ind[i]);
		mcentroids_y[i] = (mtemp_centroids_y[i] / mtemp_ind[i]);

		mtemp_centroids_x[i] = 0;
		mtemp_centroids_y[i] = 0;

		mind[i] = mtemp_ind[i];
		mtemp_ind[i] = 0;
	}
}

void launchStencilKernel (int n_samples, int n_clusters,
						  float * samples_x, float * samples_y, 
						  float * centroids_x, float * centroids_y, 
						  float * temp_centroids_x, float * temp_centroids_y, 
						  int * temp_ind, int * ind) {

	float * msamples_x, * msamples_y, * mcentroids_x, * mcentroids_y, * mtemp_centroids_x, * mtemp_centroids_y;
	int * mn_samples, * mn_clusters, * mtemp_ind, * mind;
	
	int bytes_samples = n_samples * sizeof(float), bytes_centroids = n_clusters * sizeof(float), bytes_ind = n_clusters * sizeof(int);

	cudaMalloc ((void**) &mn_samples, sizeof(int));
	cudaMalloc ((void**) &mn_clusters, sizeof(int));
	cudaMalloc ((void**) &msamples_x, bytes_samples);
	cudaMalloc ((void**) &msamples_y, bytes_samples);
	cudaMalloc ((void**) &mcentroids_x, bytes_centroids);
	cudaMalloc ((void**) &mcentroids_y, bytes_centroids);
	cudaMalloc ((void**) &mtemp_centroids_x, bytes_centroids);
	cudaMalloc ((void**) &mtemp_centroids_y, bytes_centroids);
	cudaMalloc ((void**) &mtemp_ind, bytes_ind);
	cudaMalloc ((void**) &mind, bytes_ind);
	checkCUDAError("mem allocation");

	cudaMemcpy (mn_samples, &n_samples, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (mn_clusters, &n_clusters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (msamples_x, samples_x, bytes_samples, cudaMemcpyHostToDevice);
	cudaMemcpy (msamples_y, samples_y, bytes_samples, cudaMemcpyHostToDevice);
	cudaMemcpy (mcentroids_x, centroids_x, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mcentroids_y, centroids_y, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_centroids_x, temp_centroids_x, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_centroids_y, temp_centroids_y, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_ind, temp_ind, bytes_ind, cudaMemcpyHostToDevice);
	cudaMemcpy (mind, ind, bytes_ind, cudaMemcpyHostToDevice);
	checkCUDAError("memcpy h->d");

	// launch the kernel
	startKernelTime ();
	int it, blocks = (N + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

	printf("%d\n", blocks);

	for(it=0; it<20; it++) {
		k_means <<< NUM_THREADS_PER_BLOCK, blocks >>> (mn_samples, mn_clusters, msamples_x, msamples_y, mcentroids_x, mcentroids_y, mtemp_centroids_x, mtemp_centroids_y, mtemp_ind, mind);
		updateCentroids <<<n_clusters,1>>> (mn_clusters, mcentroids_x, mcentroids_y, mtemp_centroids_x, mtemp_centroids_y, mtemp_ind, mind);
	}
	stopKernelTime ();
	checkCUDAError("kernel invocation");

	// copy the output to the host
	cudaMemcpy (centroids_x, mcentroids_x, bytes_centroids, cudaMemcpyDeviceToHost);
	cudaMemcpy (centroids_y, mcentroids_y, bytes_centroids, cudaMemcpyDeviceToHost);
	cudaMemcpy (ind, mind, bytes_ind, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy d->h");

	// free the device memory4
	cudaFree(msamples_x);
	cudaFree(msamples_y);
	cudaFree(mcentroids_x);
	cudaFree(mcentroids_y);
	cudaFree(mtemp_centroids_x);
	cudaFree(mtemp_centroids_y);
	cudaFree(mtemp_ind);
	cudaFree(mind);
	checkCUDAError("mem free");
}

void alloc() {
	samples_x = (float *) malloc(n_samples*sizeof(float));
    samples_y = (float *) malloc(n_samples*sizeof(float));

	centroids_x = (float *) malloc(n_clusters*sizeof(float));
    centroids_y = (float *) malloc(n_clusters*sizeof(float));

	temp_centroids_x = (float *) calloc(n_clusters, sizeof(float)), 
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
        centroids_x[i] = samples_x[i]; // <<cluster_i_coordenada_x>> = <<ponto_i_coordenada_x>> 
        centroids_y[i] = samples_y[i]; // <<cluster_i_coordenada_y>> = <<ponto_i_coordenada_y>>  
    } 
}

int main( int argc, char** argv) {
	n_samples = atoi(argv[1]);
	n_clusters = atoi(argv[2]);

	// ALLOC
	alloc();

	// GENERATE SAMPLES
	generate_samples();
	
	launchStencilKernel (n_samples, n_clusters, samples_x, samples_y, centroids_x, centroids_y, temp_centroids_x, temp_centroids_y, temp_ind, ind);	
	
	printf("N = %d, K = %d\n", n_samples, n_clusters);
        for(int i=0; i < n_clusters; i++) {
            printf("Center: (%.3f, %.3f) : Size %d\n", centroids_x[i], centroids_y[i], ind[i]);
        }
    printf("Iterations: %d\n", 20);

	return 0;
}