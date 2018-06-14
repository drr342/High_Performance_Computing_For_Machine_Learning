#!/bin/bash


if [ -z $1 ]; then
  echo "usage: $0 <number of ranks>"
  exit
fi

R=$1

BIN_PATH="/home/am9031/anaconda3/bin"

sbatch << EOF
#!/bin/bash
#SBATCH --job-name=lab3-$R
#SBATCH --nodes=$R
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=c32_41
#SBATCH --time=02:00:00
#SBATCH --output=out-$R.%j

module load openmpi/intel/2.0.1

echo "Job started"
#Uncomment to execute MPI code
#$BIN_PATH/mpirun -n $R hostname
$BIN_PATH/mpirun -n $R ./lab3c1
$BIN_PATH/mpirun -n $(($R-1)) ./lab3c2


#Uncomment to execute pytorch code
#$BIN_PATH/mpirun -n $R $BIN_PATH/python ./mpi_test.py
$BIN_PATH/mpirun -n $(($R-1)) $BIN_PATH/python ./lab3c3.py
$BIN_PATH/mpirun -n $R $BIN_PATH/python ./lab3c4.py
echo "Job completed"

EOF
