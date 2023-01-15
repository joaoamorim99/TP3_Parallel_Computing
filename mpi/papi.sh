#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=16
#SBATCH --partition=cpar
mpirun -np 16 ./bin/mpi_papi 10000000 4