import json
import os
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pylocogym.data.dataset import (
    Fields,
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
    StrEnum,
)

from pylocogym.data.deep_mimic_motion import (
    DeepMimicMotionDataFieldNames,
    DeepMimicMotionDataField,
    DeepMimicMotionDataSample,
    DeepMimicKeyframeMotionDataSample,
    DeepMimicMotion
)

class DeepMimicMotionCombine(MapKeyframeMotionDataset):
    """
    DeepMimic motion data.

    It combines different motion clips.
    """

    SampleType = DeepMimicKeyframeMotionDataSample

    def __init__(self, clip_paths: list, 
                 frame_transition_idx: list,
                 clips_num_repeat: list,
                 t0: float = 0.0,) -> None:
        super().__init__()

        assert all(isinstance(elem, int) and elem > 1 
                   for elem in clips_num_repeat), "Clip repeat value must be integer and greater than 1."
        
        for i in range(len(clip_paths)-1):
            motion_curr_path = os.path.join("data", "deepmimic", "motions", clip_paths[i])
            motion_next_path = os.path.join("data", "deepmimic", "motions", clip_paths[i+1])
            with open(motion_curr_path, "r") as f:
                data = json.load(f)
            curr_frames = np.array(data["Frames"])
            if i == 0:
                frames = np.vstack([curr_frames] * (clips_num_repeat[i]-1))
            else:
                frames = np.concatenate(
                    [frames, np.vstack([curr_frames] * (clips_num_repeat[i]-1))])
            
            frame_transition_idx_motion1 , frame_transition_idx_motion2 = frame_transition_idx[i]
            frames = np.concatenate([frames, curr_frames[:frame_transition_idx_motion1]])
            with open(motion_next_path, "r") as f:
                data = json.load(f)
            curr_frames = np.array(data["Frames"])
            frames = np.concatenate([frames, curr_frames[frame_transition_idx_motion2:]])
            clips_num_repeat[i+1] -= 1
        frames = np.concatenate(
            [frames, np.vstack([curr_frames] * (clips_num_repeat[-1]-1))])        

        self.dt = frames[:, 0]
        t = np.cumsum(self.dt)
        self.t = np.concatenate([[0], t])[:-1]
        self.q = frames[:, 1:]
        self.qdot = np.diff(self.q, axis=0) / self.dt[:-1, None]
        self.t0 = t0

    def __len__(self) -> int:
        # dataset length is the number of keyframes (intervals) = number of frames - 1
        return len(self.qdot)

    @property
    def duration(self) -> float:
        return self.t[-1]

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        idx = range(len(self))[idx]
        t = self.t[idx].item()
        return DeepMimicKeyframeMotionDataSample(
            dt=self.dt[idx].item(),
            t0=t + self.t0,
            q0=self.q[idx, :].copy(),
            q1=self.q[idx + 1, :].copy(),
            qdot=self.qdot[idx, :].copy(),
            phase0=t / self.duration,
            phase1=self.t[idx + 1].item() / self.duration,
        )   