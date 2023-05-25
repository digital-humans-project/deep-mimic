#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic
#SBATCH --output=./jobs/deep_mimic.out
#SBATCH --error=./jobs/deep_mimic.err
#SBATCH --gpus=1

#env2lmod
module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 eth_proxy
# activate env
# conda activate pylocoEnv
# run experiment
xvfb-run -a --server-args="-screen 0 480x480x24" /cluster/home/$USER/miniconda3/envs/pylocoEnv/bin/python src/python/main.py -wb -vr -c bob_env.json

