#define NUM_BLOCKS 128
#define NUM_THREADS_PER_BLOCK 256
#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK

int n_samples, n_clusters;
float * samples_x, *samples_y, *centroids_x, *centroids_y, *temp_centroids_x, *temp_centroids_y;

using namespace std;

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void dist_all_samples(float *samples_x, float *samples_y, float *centroids_x, float* centroids_y) {
	
	for(int i=0; i < size; i++) {

		const int idx = blockIdx.x * blockDim.x + threadIdx.x;

		/*//bounds check
		if (idx >= N) return;*/

        int cluster = 0;
        float min = 1, dist;

        for(int j=0; j < n_clusters; j++) { 

            dist = euc_dist(samples_x[idx], samples_y[idx], centroids_x[j], centroids_y[j]);
            
            if(dist<min) {
                min = dist;
                cluster = j;
            }
        }

        temp_centroids_x[cluster] += samples_x[idx];
        temp_centroids_y[cluster] += samples_y[idx];

        temp_ind[cluster]++;
    }

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}
