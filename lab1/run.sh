#!/bin/bash

#PBS -l select=2:ncpus=8:mpiprocs=8:mem=16000m,place=scatter
#PBS -l walltime=00:02:00
#PBS -m n
#PBS -o out.txt
#PBS -e err.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpirun at `which mpirun`"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./mpe_main
