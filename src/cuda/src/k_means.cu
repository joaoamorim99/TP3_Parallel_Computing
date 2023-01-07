#include "k_means.h"

#define NUM_BLOCKS 128
#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK

int n_samples, n_clusters, curr_nodo, n_nodos, * ind, * temp_ind; 

float * samples_x, * samples_y, 
      * centroids_x, * centroids_y, * part_samples_x, * part_samples_y, * temp_centroids_x, * temp_centroids_y;


using namespace std;

__device__ 
float euc_dist(float x1, float y1, float x2, float y2) {
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
}

__global__ 
void k_meansKernel (float * msamples_x, float * msamples_y, float * mcentroids_x, float * mcentroids_y, float * mtemp_centroids_x, float * mtemp_centroids_y, int * mtemp_ind) {
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

	temp_ind[cluster]++;

}


void launchStencilKernel (float * samples_x, float * samples_y, float * centroids_x, float * centroids_y, float * temp_centroids_x, float * temp_centroids_y, int temp_ind) {
	float * msamples_x, float * msamples_y, float * mcentroids_x, float * mcentroids_y, float * mtemp_centroids_x, float * mtemp_centroids_y, int * mtemp_ind;
	
	int bytes_samples = n_samples * sizeof(float), bytes_centroids = n_clusters * sizeof(float), bytes_int = n_clusters * sizeof(int);

	cudaMalloc ((void**) &msamples_x, bytes_samples);
	cudaMalloc ((void**) &msamples_y, bytes_samples);
	cudaMalloc ((void**) &mcentorids_x, bytes_centroids);
	cudaMalloc ((void**) &mcentroids_y, bytes_centroids);
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
}

/*int main( int argc, char** argv) {
	// arrays on the host
	float a[SIZE], b[SIZE], c[SIZE];

	// initialises the array
	for (unsigned i = 0; i < SIZE; ++i)
		a[i] = (float) rand() / RAND_MAX;

	stencil (a, b);
	
	launchStencilKernel (a, c);

	return 0;
}*/
