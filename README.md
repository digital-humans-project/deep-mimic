# Verify retargeting

```sh 
cmake -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
python3 src/python/main_retarget.py
```

## Changing the motion clips to verify

in main_retarget.py:

```python
dataFile = "data/deepmimic/motions/humanoid3d_walk.txt"
```

## Retarget class

```python
class Retarget:

    self.action_shape: int
    self.joint_angle_limit_low: numpy array (action_shape,)
    self.joint_angle_limit_high: numpy array (action_shape,)
    self.joint_angle_default: numpy array (action_shape,)
    self.joint_scale_factors: numpy array (action_shape,)
    
    def retarget_joint_angle(motion_clips_q, require_root = False)
        return action numpy array (action_shape)

```

### usage:
```python
retarget=Retarget(action_shape,joint_lower_limit,joint_upper_limit)

for ep in range(episodes):
    eval_env.reset()
    done = False
    t = 0
    while not done:
        eval_env.render("human")
        action = retarget.retarget_joint_angle(motion_clips_q)
        _, _, done = eval_env.step([action])
        t += 1.0/frame_rate
```