import json
import os
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import ClassVar, Dict, Literal, Optional, Tuple, Union

import numpy as np
from src.python.pylocogym.data.dataset import (
    Fields,
    KeyframeMotionDataSample,
    MapKeyframeMotionDataset,
    MotionDataSample,
    StrEnum,
)

from src.python.pylocogym.data.deep_mimic_motion import (
    DeepMimicMotionDataFieldNames,
    DeepMimicMotionDataField,
    DeepMimicMotionDataSample,
    DeepMimicKeyframeMotionDataSample,
    DeepMimicMotion
)

class DeepMimicMotionCombine(MapKeyframeMotionDataset):
    """
    Class to combines different motion clips and extract the DeepMimic data
    """

    SampleType = DeepMimicKeyframeMotionDataSample

    def __init__(self, clip_paths: list, 
                 frame_transition_idx: list,
                 clips_num_repeat: list,
                 t0: float = 0.0,) -> None:
        super().__init__()
        self.frames_exist = False
        
        '''
        Input:
            clip_paths: list of the motion names
            frame_transition_idx: list of all the tuples of the indexes at which motion-1 should transition to motion-2
            clips_num_repeat: list of number of times each motion in clips_path should repeat
        Returns:
            Extracts data from the motion_clip files and loads and initializes all the necessary data
        '''

        assert all(isinstance(elem, int) and elem > 1 
                   for elem in clips_num_repeat), "Clip repeat value must be integer and greater than 1."
        
        #looping over all the motions needed to be concatenated
        for i in range(len(clip_paths)-1):
            motion_curr_path = os.path.join("data", "deepmimic", "motions", clip_paths[i])
            motion_next_path = os.path.join("data", "deepmimic", "motions", clip_paths[i+1])
            
            with open(motion_curr_path, "r") as f:
                data = json.load(f)
            curr_frames = np.array(data["Frames"])
            
            for rep in range(clips_num_repeat[i]-1):
                if not self.frames_exist:
                    #initializing frames here
                    frames = curr_frames
                    frames[-1,0] = frames[-2, 0]
                    rep +=1
                    self.frames_exist = True
                    continue

                #looping over the current motion n-1 times and appending to final frames
                trans_frames = self.track_root_and_time(curr_frames, frames[-1][1], frames[-1][3])
                frames = np.concatenate([frames, trans_frames]) 
            
            #extracting the frame number at which current motion-1 transitions into next motion-2
            frame_transition_idx_motion1 , frame_transition_idx_motion2 = frame_transition_idx[i]
            
            #appending only the number of frames of current motion till the transition point
            trans_frames = self.track_root_and_time(curr_frames, frames[-1][1], frames[-1][3], 
                                                    frame_transition_idx_motion1, "first")
            frames = np.concatenate([frames, trans_frames[:frame_transition_idx_motion1+1]])

            #reading and appending only onwards the frame number of motion-2 where transition occurs
            with open(motion_next_path, "r") as f:
                data = json.load(f)
            curr_frames = np.array(data["Frames"])
            trans_frames = self.track_root_and_time(curr_frames, frames[-1][1], frames[-1][3], 
                                                    frame_transition_idx_motion2, "second")
            frames = np.concatenate([frames, trans_frames[frame_transition_idx_motion2:]])
            clips_num_repeat[i+1] -= 1
        
        #appending the remaining repeats of the last motion
        for rep in range(clips_num_repeat[-1]):
            trans_frames = self.track_root_and_time(curr_frames, frames[-1][1], frames[-1][3])
            frames = np.concatenate([frames, trans_frames])

        #dumping data into json file
        json_save_path =  "".join([name.replace("humanoid3d_","").replace(".txt","") 
                                   for name in clip_paths])+str("_multiclip.txt")
        self.create_new_datafile("none", frames.copy(), json_save_path)

        #extracting and initializing information related to time, q, qdot    
        self.dt = frames[:, 0]
        t = np.cumsum(self.dt)
        self.t = np.concatenate([[0], t])[:-1]
        self.q = frames[:, 1:]
        self.qdot = np.diff(self.q, axis=0) / self.dt[:-1, None]
        self.t0 = t0

    def track_root_and_time(self, curr_frames, frames_last_state_x, frames_last_state_z, 
                            frame_transition_idx=0, frame_transition_loc="full"):
        '''
        Input:
            curr_frames: frames extracted from the motion clip data
            frames_last_state_x, frames_last_state_z: xz root position values of the previous 
                    frame to which curr_frames will be appended
            frame_transition_idx: index value of the frame at which the transition of motion should occur
            frame_transition_loc: string indicating how the motion will be appending so that the
                    root tracking can be continued accordingly
        Returns:
            trans_frame: the processed frames that need to be appended to the current frames
        '''

        trans_frames = curr_frames.copy()
        trans_frames[-1,0] = trans_frames[-2,0] #to avoid divided by zero error caused due to last time frame of the motion being 0
        
        #updating root values based on transition type
        if frame_transition_loc == "full": #entire current motion is repeating
            trans_frames[:,1] += frames_last_state_x
            trans_frames[:,3] += frames_last_state_z
        elif frame_transition_loc == "first": #current motion is going to transition into next
            trans_frames[:frame_transition_idx+1, 1] += frames_last_state_x
            trans_frames[:frame_transition_idx+1, 3] += frames_last_state_z
        elif frame_transition_loc == "second": #current motion is being transitioned into from previous
            trans_frames[frame_transition_idx:, 1] += frames_last_state_x
            trans_frames[frame_transition_idx:, 3] += frames_last_state_z
        
        return trans_frames
    
    def create_new_datafile(self, loop, frames, json_save_path):
 
        save_path = os.path.join('data/deepmimic/motions/', json_save_path)
        data = {
            "Loop": loop,
            "Frames": frames
        }
        with open(save_path, 'w') as file:
            json.dump(data, file)
        
    def __len__(self) -> int:
        # dataset length is the number of keyframes (intervals) = number of frames - 1
        return len(self.qdot)

    @property
    def duration(self) -> float:
        #returns the total time of the motion
        return self.t[-1]

    def __getitem__(self, idx) -> DeepMimicKeyframeMotionDataSample:
        #to get a Data Sample when provided with idx
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