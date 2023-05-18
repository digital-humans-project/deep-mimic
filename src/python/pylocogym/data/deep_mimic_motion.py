import json
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pylocogym.data.dataset import (
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
    StrEnum,
)


class DeepMimicMotionDataFields(StrEnum):
    """
    Enum class for DeepMimic motion data field names.
    """

    ROOT_POS = auto()
    ROOT_ROT = auto()
    CHEST_ROT = auto()
    NECK_ROT = auto()
    R_HIP_ROT = auto()
    R_KNEE_ROT = auto()
    R_ANKLE_ROT = auto()
    R_SHOULDER_ROT = auto()
    R_ELBOW_ROT = auto()
    L_HIP_ROT = auto()
    L_KNEE_ROT = auto()
    L_ANKLE_ROT = auto()
    L_SHOULDER_ROT = auto()
    L_ELBOW_ROT = auto()


_fields: Dict[DeepMimicMotionDataFields, Tuple[int, int]] = {
    DeepMimicMotionDataFields.ROOT_POS: (0, 3),
    DeepMimicMotionDataFields.ROOT_ROT: (3, 7),
    DeepMimicMotionDataFields.CHEST_ROT: (7, 11),
    DeepMimicMotionDataFields.NECK_ROT: (11, 15),
    DeepMimicMotionDataFields.R_HIP_ROT: (15, 19),
    DeepMimicMotionDataFields.R_KNEE_ROT: (19, 20),
    DeepMimicMotionDataFields.R_ANKLE_ROT: (20, 24),
    DeepMimicMotionDataFields.R_SHOULDER_ROT: (24, 28),
    DeepMimicMotionDataFields.R_ELBOW_ROT: (28, 29),
    DeepMimicMotionDataFields.L_HIP_ROT: (29, 33),
    DeepMimicMotionDataFields.L_KNEE_ROT: (33, 34),
    DeepMimicMotionDataFields.L_ANKLE_ROT: (34, 38),
    DeepMimicMotionDataFields.L_SHOULDER_ROT: (38, 42),
    DeepMimicMotionDataFields.L_ELBOW_ROT: (42, 43),
}


@dataclass
class DeepMimicMotionDataSample(MotionDataSample):
    Fields: ClassVar = DeepMimicMotionDataFields
    fields: ClassVar = _fields  # type: ignore


@dataclass
class DeepMimicKeyframeMotionDataSample(KeyframeMotionDataSample):
    Fields: ClassVar = DeepMimicMotionDataFields
    fields: ClassVar = _fields  # type: ignore
    BaseSampleType: ClassVar = DeepMimicMotionDataSample


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
        # TODO: last frame is dropped here because we don't have qdot for it
        return len(self.qdot)

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        idx = np.clip(idx, 0, len(self) - 1)
        return DeepMimicKeyframeMotionDataSample(
            dt=self.dt[idx].item(),
            t=self.t[idx].item(),
            q=self.q[idx, :].copy(),
            qdot=self.qdot[idx, :].copy(),
        )
