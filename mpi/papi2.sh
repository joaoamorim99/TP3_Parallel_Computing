#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=8
#SBATCH --partition=cpar
mpirun -np 8 ./bin/omp_papi 10000000 4 32