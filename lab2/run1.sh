#!/bin/bash

#PBS -l select=1:ncpus=12:ompthreads=12:mem=5000m
#PBS -l walltime=00:01:00
#PBS -m n
#PBS -o out1.txt
#PBS -e err1.txt

cd $PBS_O_WORKDIR

echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo
./main1
