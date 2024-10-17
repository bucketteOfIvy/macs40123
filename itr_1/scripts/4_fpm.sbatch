#!/bin/bash

#SBATCH --job-name=wimer_fpm_itr
#SBATCH --output=4_fpm.out
#SBATCH --error=4_fpm.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=40G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 1 --executor-memory 5G --driver-memory 5G ./4_fpm.py