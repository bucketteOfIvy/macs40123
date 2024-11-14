#!/bin/bash
#SBATCH --job-name=4_run_node2vec.py
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=macs40123
#SBATCH --mem-per-cpu=30G

module load python cuda
pip install pyogrio pyarrow

python 4_run_node2vec.py
