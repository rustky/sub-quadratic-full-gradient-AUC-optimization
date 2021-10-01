#!/bin/bash
#SBATCH --job-name=torch_test
#SBATCH --time=20:00
#SBATCH --output=/projects/genomic-ml/sub-quadratic-full-gradient-AUC-optimization/app/driver.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
source ~/.bashrc
conda activate emacs1
cd /projects/genomic-ml/sub-quadratic-full-gradient-AUC-optimization/app
python driver.py
