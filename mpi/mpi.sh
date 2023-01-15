#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=16
#SBATCH --partition=cpar
mpirun -np 16 ./bin/k_means 10000000 4