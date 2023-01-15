#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=8
#SBATCH --partition=cpar
mpirun -np 8 ./bin/k_means_omp 10000000 4 32