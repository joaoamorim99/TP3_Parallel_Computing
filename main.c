#include "k_means.h"

#define n_samples 1000
#define n_clusters 4
#define n_threads 2


int main(int argc, char *argv[]) {
    int n_nodos, curr_nodo;
    
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_nodo); // gets this process rank
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodos); // gets number of ranks

    Nodo nodos[n_nodos];

    for(int i=0; i<n_nodos; i++) {
        initNodo(nodos[i], n_samples, n_clusters/n_nodos);
    }
    
    /* Process 0 sends and Process 1 receives 
    if (rank == 0) {
        msg = 123456;
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } // (buf, count, datatype, dest, tag, comm)
    
    else if (rank == 1) {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        printf("Received %d\n", msg);  
    }
    
    MPI_Finalize();*/
    return 0;

}