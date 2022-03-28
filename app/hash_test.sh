#!/bin/bash
#SBATCH --job-name=hash_test                        # the name of your job
#SBATCH --output=/projects/genomic-ml/sub-quadratic-full-gradient-AUC-optimization/app/results/test/error.txt
#SBATCH --time=120:00                            # 20 min, shorter time, quicker start, max run time
#SBATCH --mem=32000                              # 2GB of memory
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL


# Run your application: precede the application command with 'srun'
# A couple example applications...

python3 temp_driver.py 10 0.1 square_hinge_test .0001 ./results/test CAT_VS_DOG ResNet20 SGD 5
