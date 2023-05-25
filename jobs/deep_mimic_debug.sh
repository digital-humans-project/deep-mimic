#!/bin/bash

#SBATCH -n 4                              # Number of cores
#SBATCH --time=00:10:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic_debug
#SBATCH --output=./jobs/deep_mimic_debug.out
#SBATCH --error=./jobs/deep_mimic_debug.err
#SBATCH --gpus=1

source scripts/setup.sh

xvfb-run -a --server-args="-screen 0 480x480x24" python src/python/main.py -wb -vr -c bob_env_debug.json -d
