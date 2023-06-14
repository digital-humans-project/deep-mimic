#!/bin/bash

#SBATCH -n 20                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic_walkpunch
#SBATCH --output=./out/deep_mimic_walkpunch.out
#SBATCH --error=./out/deep_mimic_walkpunch.err
#SBATCH --gpus=1

xvfb-run -a --server-args="-screen 0 480x480x24" python -u src/python/main.py -wb -vr -c bob_env_multi_walkpunch.json