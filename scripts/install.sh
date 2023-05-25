#!/bin/sh

module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 glfw/3.3.4 eth_proxy
python -m venv venv

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# patch wandb to fix the incompatible version of gym
patch venv/lib64/python3.9/site-packages/wandb/integration/gym/__init__.py scripts/patch_wandb_0_15_3_gym_0_21_0.patch

cmake -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build