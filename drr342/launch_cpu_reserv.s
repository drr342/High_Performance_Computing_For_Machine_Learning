#!/bin/bash

#SBATCH --job-name=lab1_drr342_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=62GB
#SBATCH --time=00:30:00
#SBATCH --res=morari 
#SBATCH --partition=c26
#SBATCH --exclusive
#SBATCH --output=lab1_drr342_cpu_%j.out

module purge
module load intel/17.0.1
./lab1

module purge
module load python3/intel/3.6.3
module load numpy/python3.6/intel/1.14.0
module load pytorch/python3.6/0.3.0_4
python ./lab1.py
python ./lab1.pytorch 5

#The argument passed to lab1.pytorch is the number of workers (if no argument is passed it defaults to 1)

