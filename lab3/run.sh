#!/bin/bash

#PBS -l select=1:ncpus=2:mpiprocs=2:mem=6000m,place=scatter
#PBS -l walltime=00:01:30
#PBS -m n
#PBS -o out.txt
#PBS -e err.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./matrix_multiplying
