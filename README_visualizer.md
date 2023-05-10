# Simple visualizer

```sh 
cmake -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
python3 src/python/main_visualizer.py
```

## Joint coordinate retarget table

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