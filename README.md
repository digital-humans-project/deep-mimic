# Deep Mimic with Pyloco Environment

This repository contains a fork of Pyloco's implementation of the humanoid model used in DeepMimic, and our implementation of the RL pipeline to mimic motion from a reference motion clip on the humanoid model. Related paper links: https://xxxxxxxx

## Install

Install PyTorch using conda or pip.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then install the package.

```bash
pip install --upgrade pip
pip install -r requirements.txt 
```

## Play the motion clips

```bash
cd deep-mimic
python src/python/main_retarget.py
```
To choose the specific motion you want to play, need to change the clip file inside `src/python/main.py`, for example:
```sh 
motion_clip_file = "humanoid3d_walk.txt"
config = "bob_env_walk.json"
```

## Train the network (on Euler)

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
  $ sbatch ./jobs/deep_mimic_walk.sh
  ```
### Change config

- config in `data/conf/bob_env_walk.json`
```json
"env_id": "PylocoVanilla-v0", // can choose from {"PylocoVanillaTask-v0","ResidualEnv-v0","ResidualEnv-v1","PylocoMultiClip-v0"}, determine the method to use
"motion_file": "humanoid3d_walk.txt", // Determine motion clips to mimic
"train_hyp_params": {...}, 
"environment_params":{...},
"reward_params":{...},
"model_params": {...},
```

## Test the network
```bash
cd deep-mimic
python src/python/main_test_model.py
```
To choose the pre-trained model file, need to change it inside `src/python/main_test_model.py`, for example:
```sh 
model_file = "model_data/walk/model_34000000_steps.zip"
```


## Visulaizer for retarget 

```bash
cd deep-mimic
python src/python/main_visualizer.py
```

### Joint coordinate retarget table

|  joint in bob   | index in action | joint in motions clip | relationship |
|  :----:         | :----: | :----:                | :----:|
|  lowerback_x    | 0      | chest_x               | prime |
|  lowerback_y    | 3      | chest_y               | prime |
|  lowerback_z    | 6      | chest_z               | prime |
|  upperback_x    | 9      | chest_x               | subordinate |
|  upperback_y    | 12     | chest_y               | subordinate |
|  upperback_z    | 15     | chest_z               | subordinate |
|  lowerneck_x    | 18     | neck_x                | prime |
|  lowerneck_y    | 23     | neck_y                | prime |
|  lowerneck_z    | 26     | neck_z                | prime |
|  upperneck_x    | 29     | neck_z                | subordinate |
|  upperneck_y    | 32     | neck_y                | subordinate |
|  upperneck_z    | 35     | neck_z                | subordinate |
|  lScapula_y     | 19     | left shoulder_y       | subordinate |
|  lScapula_z     | 24     | left shoulder_z       | subordinate |
|  lShoulder_1    | 27     | left shoulder_x       | prime      |
|  lShoulder_2    | 30     | left shoulder_z       | prime      |
|lShoulder_torsion| 33     | left shoulder_y       | prime      |
|lElbow_flexion_extension|36|left elbow            | prime      |
|  lElbow_torsion | 38     | left elbow(none)      | additional  |
|  lWrist_x       | 40     | left wrist(none)      | additional  |
|  lWrist_z       | 42     | left wrist(none)      | additional  |
|  rScapula_y     | 20     | right shoulder_y      | subordinate |
|  rScapula_z     | 25     | right shoulder_z      | subordinate |
|  rShoulder_1    | 28     | right shoulder_x      | prime      |
|  rShoulder_2    | 31     | right shoulder_z      | prime      |
|rShoulder_torsion| 34     | right shoulder_y      | prime      |
|rElbow_flexion_extension|37|right elbow           | prime      |
|  rElbow_torsion | 39     | right elbow(none)     | additional  |
|  rWrist_x       | 41     | right wrist(none)     | additional  |
|  rWrist_z       | 43     | right wrist(none)     | additional  |
|  lHip_1         | 1      | left hip_x            | prime      |
|  lHip_2         | 4      | left hip_z            | prime      |
|  lHip_torsion   | 7      | left hip_y            | prime      |
|  lKnee          | 10     | left knee             | prime      |
|  lAnkle_1       | 13     | left ankle_x          | prime      |
|  lAnkle_2       | 16     | left ankle_z          | prime      |
|  ***none***     | --     | left ankle_y          | prime      |
|  lToeJoint      | 21     | left toe(none)        | additional  |
|  rHip_1         | 2      | right hip_x           | prime      |
|  rHip_2         | 5      | right hip_z           | prime      |
|  rHip_torsion   | 8      | right hip_y           | prime      |
|  rKnee          | 11     | right knee            | prime      |
|  rAnkle_1       | 14     | right ankle_x         | prime      |
|  rAnkle_2       | 17     | right ankle_z         | prime      |
|  ***none***     | --     | right ankle_y         | prime      |
|  rToeJoint      | 22     | right toe(none)       | additional  |
