CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
THREADS = 32
CFLAGS = -O2 -fopenmp
.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c 
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)$(EXEC)

clean:
	rm -r bin/*

perfseq:
	perf stat -e instructions,cycles -r 5 $(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

perfpar:
	perf stat -e instructions,cycles -r 5 $(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)
	
