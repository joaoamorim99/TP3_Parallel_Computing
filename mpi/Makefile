CC = mpicc
BIN = bin/
SRC = src/
INCLUDES = include/
MPIRUN = mpirun
EXEC = k_means
EXEC2 = k_means_omp
THREADS = 4
NPROC = 4
CFLAGS = -O2 -fopenmp -lm
.DEFAULT_GOAL = k_means

k_means: $(SRC)mpi.c 
	$(CC) $(CFLAGS) $(SRC)mpi.c -o $(BIN)$(EXEC)
	$(CC) $(CFLAGS) $(SRC)mpi.c -o $(BIN)$(EXEC2)

clean:
	rm -r bin/*

perfmpi:
	perf stat -e instructions,cycles -r 5 $(MPIRUN) -np $(NPROC) $(BIN)$(EXEC) 10000000 4

perfomp:
	perf stat -e instructions,cycles -r 5 $(MPIRUN) -np $(NPROC) ./$(BIN)$(EXEC) 10000000 4 $(THREADS)

runseq:	
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)

runmpi:
	$(MPIRUN) -np $(NPROC) ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runompmpi:
	$(MPIRUN) -np $(NPROC) ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)
	
