import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from pylocogym.data.dataset import (
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
)

_fields = {
    "root_pos": (0, 3),
    "root_rot": (3, 7),
    "chest_rot": (7, 11),
    "neck_rot": (11, 15),
    "r_hip_rot": (15, 19),
    "r_knee_rot": (19, 20),
    "r_ankle_rot": (20, 24),
    "r_shoulder_rot": (24, 28),
    "r_elbow_rot": (28, 29),
    "l_hip_rot": (29, 33),
    "l_knee_rot": (33, 34),
    "l_ankle_rot": (34, 38),
    "l_shoulder_rot": (38, 42),
    "l_elbow_rot": (42, 43),
}


@dataclass
class DeepMimicMotionDataSample(MotionDataSample):
    fields = _fields


@dataclass
class DeepMimicKeyframeMotionDataSample(KeyframeMotionDataSample):
    fields = _fields
    BaseSampleType = DeepMimicMotionDataSample


class DeepMimicMotion(MapKeyframeMotionDataset):
    SampleType = DeepMimicKeyframeMotionDataSample

    def __init__(self, path: Union[str, Path], t0: float = 0.0, loop: Optional[Literal["wrap", "none"]] = None) -> None:
        super().__init__()
        with open(path, "r") as f:
            data = json.load(f)
        self.loop = data["Loop"] if loop is None else loop
        assert self.loop in ["wrap", "none"]
        self.frames = np.array(data["Frames"])
        self.dt = self.frames[:, 0]
        if self.loop == "wrap":
            t = np.cumsum(np.concatenate([self.dt, self.dt[-2::-1]]))
        else:
            t = np.cumsum(self.dt)
        self.t = np.concatenate([[0], t]) + t0

    @property
    def raw_len(self) -> int:
        return len(self.frames)

    def __len__(self) -> int:
        return self.raw_len if self.loop == "none" else 2 * self.raw_len - 1

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        wrap_idx = idx
        if self.loop == "wrap" and idx >= self.raw_len:
            wrap_idx = -(idx % self.raw_len + 2)

        return DeepMimicKeyframeMotionDataSample(
            dt=self.dt[wrap_idx],
            t=self.t[idx],
            q=self.frames[wrap_idx, 1:],
        )
