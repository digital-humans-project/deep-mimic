import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from pylocogym.data.keyframe_dataset import KeyframeMotionDataSample, MapKeyframeMotionDataset


@dataclass
class DeepMimicMotionDataSample(KeyframeMotionDataSample):
    fields: Dict = field(default_factory=lambda: {
        'root_pos': (0, 3),
        'root_rot': (3, 7),
        'chest_rot': (7, 11),
        'neck_rot': (11, 15),
        'r_hip_rot': (15, 19),
        'r_knee_rot': (19, 23),
        'r_ankle_rot': (23, 27),
        'r_shoulder_rot': (27, 31),
        'r_elbow_rot': (31, 35),
        'l_hip_rot': (35, 39),
        'l_knee_rot': (39, 43),
        'l_ankle_rot': (43, 47),
        'l_shoulder_rot': (47, 51),
        'l_elbow_rot': (51, 55),
    })


class DeepMimicMotion(MapKeyframeMotionDataset):
    def __init__(self, path) -> None:
        super().__init__()
        with open(path, 'r') as f:
            data = json.load(f)
        self.loop = data['Loop']
        assert self.loop in ['wrap', 'none']
        self.frames = np.array(data['Frames'])
        self.dt = self.frames[:, 0]
        self.t = np.cumsum(self.dt)

    def __len__(self) -> int:
        return len(self.t) if self.loop == 'none' else 2 * len(self.t) - 1

    def __getitem__(self, idx) -> DeepMimicMotionDataSample:
        if self.loop == 'wrap' and idx >= len(self.t):
            idx = -(idx % len(self.t) + 2)

        return DeepMimicMotionDataSample(
            dt=self.dt[idx],
            t=self.t[idx],
            q=self.frames[idx, 1:],
        )
