CC = mpicc
BIN = bin/
SRC = src/
INCLUDES = include/
MPIRUN = mpirun
EXEC = k_means
EXEC2 = k_means_omp
EXEC_PAPI = mpi_papi
EXEC_PAPI2 = omp_papi
THREADS = 4
NPROC = 3
CFLAGS = -O2 
CFLAGS_OMP = -fopenmp -lm
CFLAGS_PAPI = -O2 -Wall -fno-omit-frame-pointer -g -I/share/apps/papi/5.4.1/include -L/share/apps/papi/5.4.1/lib -lpapi
CFLAGS_PAPI_OMP = -O2  -fopenmp -lm -Wall -fno-omit-frame-pointer -g -I/share/apps/papi/5.4.1/include -L/share/apps/papi/5.4.1/lib -lpapi
.DEFAULT_GOAL = k_means

k_means: 
	$(CC) $(CFLAGS) $(SRC)mpi.c -o $(BIN)$(EXEC)
	$(CC) $(CFLAGS) $(CFLAGS_OMP) $(SRC)omp+mpi.c -o $(BIN)$(EXEC2)

k_means_papi:
	$(CC) $(CFLAGS_PAPI) $(SRC)mpi_papi.c -o $(BIN)$(EXEC_PAPI)
	$(CC) $(CFLAGS_PAPI_OMP) $(SRC)omp+mpi_papi.c -o $(BIN)$(EXEC_PAPI2)

clean:
	rm -r bin/*

perfmpi:
	perf stat -e instructions,cycles -r 5 $(MPIRUN) -np $(NPROC) $(BIN)$(EXEC) 10000000 4

perfomp:
	perf stat -e instructions,cycles -r 5 $(MPIRUN) -np $(NPROC) ./$(BIN)$(EXEC) 10000000 4 $(THREADS)

runmpi:
	sbatch mpi.sh

runompmpi:
	sbatch omp.sh
	
runpapi:
	sbatch papi.sh

runpapi2:
	sbatch papi2.sh
