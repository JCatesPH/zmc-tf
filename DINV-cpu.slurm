#!/bin/bash


#SBATCH -J zmc-tf # name of the job

### set output files
#SBATCH -o ./tf-device.%A.out #output file 
#SBATCH -e ./tf-device.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos main
#SBATCH -p main
#SBATCH --mem-per-cpu 5g

# run application
srun python ./DINV.py
