# Leveraging Motion Imitation in RL for Biped Character

## Build

Setup the Python environment either with conda or virtualenv. Then build the C++ extention with cmake:

```sh
cmake -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
```

Install the requirements with pip:

```sh
cd src/python
pip install -e .
```

## Dataset

### Base Keyframe Dataset

Base datasets include iterable-style and map-style dataset interfaces. Each dataset allows to iterate over the samples and returns a `KeyframeMotionDataSample` object, which includes the following fields:

```python
@dataclass
class KeyframeMotionDataSample:
    t: float
    q: np.ndarray
    dt: float
```

### Deep Mimic Dataset

Deep Mimic dataset is an infinite iterable-style dataset that keeps sampling from a group of motions, as specified in the JSON file in the `datasets` folder. The dataset returns a `DeepMimicKeyframeMotionDataSample` object, which includes the fields in `KeyframeMotionDataSample` and the following additional fields:

```python
_fields = {
  "root_pos": q[0:3],
  "root_rot": q[3:7],
  "chest_rot": q[7:11],
  "neck_rot": q[11:15],
  "r_hip_rot": q[15:19],
  "r_knee_rot": q[19:20],
  "r_ankle_rot": q[20:24],
  "r_shoulder_rot": q[24:28],
  "r_elbow_rot": q[28:29],
  "l_hip_rot": q[29:33],
  "l_knee_rot": q[33:34],
  "l_ankle_rot": q[34:38],
  "l_shoulder_rot": q[38:42],
  "l_elbow_rot": q[42:43]
}
```

### Linear Interpolation Continuous Dataset (Lerp Dataset)

Lerp dataset allows to evaluate motion at a certain time $t$ given a keyframe dataset, and the motion will be linearly interpolated between the two nearest keyframes. The usage is simple as follows:

```python
dataset = DeepMimicMotionDataset("data/deepmimic/datasets/humanoid3d_clips_locomotion.txt")
lerp = LerpMotionDataset(dataset)

for t in np.linspace(0, 10, 1000):
    sample = lerp.eval(t)
    if sample is None: # t is out of range
        break
    print(sample.t, sample.q)
```
