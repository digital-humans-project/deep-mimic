#!/bin/bash

#SBATCH -n 20                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic_residual_data
#SBATCH --output=./jobs/deep_mimic_residual_data.out
#SBATCH --error=./jobs/deep_mimic_residual_data.err
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

xvfb-run -a --server-args="-screen 0 480x480x24" python src/python/main.py -wb -vr -c bob_env_walk_residual_data.json
