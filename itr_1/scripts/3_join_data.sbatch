#!/bin/bash

#SBATCH --job-name=spatial_join
#SBATCH --output=spatial_join.out
#SBATCH --error=spatial_join.err
#SBATCH --ntasks=1
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --mem=10G

module load python

pip install geopandas

python ./3_join_data.py
