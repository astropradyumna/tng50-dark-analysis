#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --mem=300gb
#SBATCH -o output_log/t50h.out
#SBATCH -e output_log/t50h.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng50-dark-analysis
python3 tng50hydro_data.py 2
# python3 tng50dm_data.py 2
# for ((i = 0; i <= 2; i++))
# do
#     # python3 extract_particle_data_within_3rvir.py $i
#     python3 tng50hydro_data.py $i
# done