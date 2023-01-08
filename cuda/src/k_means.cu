#include "k_means.h"

#define NUM_BLOCKS 128
#define NUM_THREADS_PER_BLOCK 256

using namespace std;

float * samples_x, * samples_y, * centroids_x, * centroids_y, * temp_centroids_x, * temp_centroids_y;
int * temp_ind, * ind;

__const__ n_samples, n_clusters;

__device__ float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

// KERNEL
__global__ void dist_all_samples (float * msamples_x, float * msamples_y, float * mtemp_centroids_x, float * mtemp_centroids_y, int * mtemp_ind) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int cluster = 0;
	float dist=1;

	for(int i=0; i<n_clusters; i++) {
		
		dist = euc_dist(msamples_x[id], msamples_y[id], mcentroids_x[i], mcentroids_y[i]);
            
		if(dist<min) {
			min = dist;
			cluster = j;
		}
	}

	mtemp_centroids_x[cluster] += msamples_x[id];
	mtemp_centroids_y[cluster] += msamples_y[id];

	mtemp_ind[cluster]++;
}


/*void launchStencilKernel (float * samples_x, float * samples_y, float * temp_centroids_x, float * temp_centroids_y, int temp_ind) {
	float * msamples_x, float * msamples_y, float * mcentroids_x, float * mcentroids_y, float * mtemp_centroids_x, float * mtemp_centroids_y, int * mtemp_ind;
	
	int bytes_samples = n_samples * sizeof(float), bytes_centroids = n_clusters * sizeof(float), bytes_int = n_clusters * sizeof(int);

	cudaMalloc ((void**) &msamples_x, bytes_samples);
	cudaMalloc ((void**) &msamples_y, bytes_samples);
	cudaMalloc ((void**) &mtemp_centroids_x, bytes_centroids);
	cudaMalloc ((void**) &mtemp_centroids_y, bytes_centroids);
	cudaMalloc ((void**) &mtemp_ind, bytes_int);
	checkCUDAError("mem allocation");

	cudaMemcpy (msamples_x, samples_x, bytes_samples, cudaMemcpyHostToDevice);
	cudaMemcpy (msamples_y, samples_y, bytes_samples, cudaMemcpyHostToDevice);
	cudaMemcpy (mcentroids_x, centroids_x, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mcentroids_y, centroids_y, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_centroids_x, temp_centroids_x, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_centroids_y, temp_centroids_y, bytes_centroids, cudaMemcpyHostToDevice);
	cudaMemcpy (mtemp_ind, temp_ind, bytes_int, cudaMemcpyHostToDevice);
	checkCUDAError("memcpy h->d");

	// launch the kernel
	startKernelTime ();
	for(int i=0; i<20; i++)  {
		k_meansKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS >>> (da, dc);
	}
	stopKernelTime ();
	checkCUDAError("kernel invocation");

	// copy the output to the host
	cudaMemcpy (c,dc,bytes,cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy d->h");

	// free the device memory
	cudaFree(da);
	cudaFree(dc);
	checkCUDAError("mem free");
}*/

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
	/*n_samples = argv[1];
	n_clusters = argv[2];*/
	n_samples = 100;
	n_clusters = 4;

	// ALLOC
	samples_x = (float *) malloc(n_samples*sizeof(float));
    samples_y = (float *) malloc(n_samples*sizeof(float));

	centroids_x = (float *) malloc(n_clusters*sizeof(float));
    centroids_y = (float *) malloc(n_clusters*sizeof(float));

	temp_centroids_x = (float *) calloc(n_clusters, sizeof(float));
    temp_centroids_y = (float *) calloc(n_clusters, sizeof(float));
    
    ind = (int *) calloc(n_clusters, sizeof(int));
    temp_ind = (int *) calloc(n_clusters, sizeof(int));

	// GENERATE SAMPLES
	generate_samples();
	

	/*stencil (a, b);
	
	launchStencilKernel (a, c);*/

	return 0;
}
