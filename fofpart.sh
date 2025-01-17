#!/bin/bash -l
#SBATCH -J FoFPart
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
#SBATCH --mem=100gb
###SBATCH --mem-per-cpu=G
#SBATCH --time=48:00:00
#SBATCH -o output_log/fofpart.out
#SBATCH -e output_log/fofpart.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL
###SBATCH --nodelist=r14

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng50-dark-analysis

for i in 2
do
    python3 extract_particle_data_within_3rvir.py $i
done