#!/bin/bash

#SBATCH --job-name=lab1_drr342_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=250GB
#SBATCH --gres=gpu:k80:4
#SBATCH --time=00:30:00
#SBATCH --res=morari 
#SBATCH --partition=k80_4
#SBATCH --exclusive
#SBATCH --output=lab1_drr342_gpu_%j.out

module purge
module load python3/intel/3.6.3
module load numpy/python3.6/intel/1.14.0
module load pytorch/python3.6/0.3.0_4
python ./lab1.pytorch 5

#The argument passed to lab1.pytorch is the number of workers (if no argument is passed it defaults to 1)
