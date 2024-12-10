#!/bin/bash

#SBATCH --job-name=fewer_dimensions_please
#SBATCH --output=8_out.out
#SBATCH --error=8_err.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=40G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 8 --executor-memory 5G 8_pcas.py


