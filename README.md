# Naive version for training in euler

noted: an ugly code version, just for testing avaibility of whole thing

------------------------------------------------


## STEP 1: Decided which motions clip to learn

- go finds `src/python/main.py` line 49:
```sh 
motion_clip_file = "humanoid3d_walk.txt"
```
- change the clip file to whatever you want

------------------------------------------------

## STEP 2: Reward function part

- in `src/python/pylocogym/envs/rewards/bob/humanoid_reward.py`:

```python
def compute_reward():
    inplements 6 reward terms

    ## Those for robot's root
    1. root_height reward, (make the root y pos mimic data desired height)
    2. forward_vel_reward (compared for each time_steps, the difference between the root x&z position and that from our model)
    3. root_ori_reward (make the root ori mimic data desired ori)
    

    ## Those for robot's joints
    4. smoothness reward (make the useless joints move smoothly)
    5. joint_reward (mimic the motion clips)
    6. joint_vel_reward (mimic the motion clips)
    7. end_effector_reward (mimic the motion clips)
```

- noted: these designs are merely naive versions, only for the walking task,
feel free to modify them to proper way

------------------------------------------------

## STEP 3: Reward function parameters
- go finds `data/conf/bob_env.json`:
- can change weight and sigma, also the network structure

------------------------------------------------
## STEP 4: Slow Down the motion clips (optional)
- in `data/conf/bob_env.json` line 30:
```python
 "clips_play_speed": 0.5,
```
- this parameter is for slowing down the motion clips 0.5 time
- the reason to do that is the motion clips of walking are too short, only 1.1 seconds. if you want to keep the original motion clips, just set `clips_play_speed` = 1

------------------------------------------------

## STEP 4: Run training in euler
- almost the same procedure as in assignment 2

```sh 
$ conda activate pylocoEnv  
$ cd deep-mimic (clone the repo)
$ git checkout naive_train
$ pip install -r requirements.txt 
$ wandb login
```

### Compile on server
- Build `pyloco.so` (python wrapper of pyloco C++ libraries)
  ```sh
  $ mkdir build && cd build   
  $ env2lmod
  $ module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 eth_proxy
  # IMPORTANT: If you use a conda environment or a virtual environment, you should add 
  # -DPython_EXECUTABLE=<PYTHON INTERPRETER PATH>  (Use absolute paths) e.g.:
  # cmake -DPython_EXECUTABLE=/cluster/home/mimora/miniconda3/envs/pylocoEnv/bin/python3 -DCMAKE_BUILD_TYPE=Release ../
  $ cmake -DPython_EXECUTABLE=/cluster/home/<YOUR_USERNAME>/miniconda3/envs/pylocoEnv/bin/python3 -DCMAKE_BUILD_TYPE=Release ../
  $ make 
  # Return to repo folder
  $ cd ..
  ```

  ### Run jobs on server
 
- Run
  ```sh
  # Before you start a job, make sure to run the following two commands, every time you start a new ssh connection to Euler.
  $ env2lmod
  $ module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 eth_proxy  
  # Submit job
  $ sbatch ./jobs/deep_mimic
  ```