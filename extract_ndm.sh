#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH -o output_log/t50ndm.out
#SBATCH -e output_log/t50ndm.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng50-dark-analysis

for ((i = 0; i <= 2; i++))
do
    python3 extract_ndm.py $i
done