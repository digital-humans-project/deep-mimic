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
        assert self.loop in ["wrap", "none", "mirror"]

        frames = np.array(data["Frames"])
        if self.loop == "mirror":
            frames = np.concatenate([frames, frames[-2::-1]])

        self.dt = frames[:, 0]
        t = np.cumsum(self.dt)
        self.t = np.concatenate([[0], t]) + t0
        self.q = frames[:, 1:]
        self.qdot = np.diff(self.q, axis=0) / self.dt[:-1, None]

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        idx = np.clip(idx, 0, len(self) - 1)
        return DeepMimicKeyframeMotionDataSample(
            dt=self.dt[idx],
            t=self.t[idx],
            q=self.q[idx, :],
            qdot=self.qdot[idx, :],
        )
