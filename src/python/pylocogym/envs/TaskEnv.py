import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from typing import List

import numpy as np
from gym import spaces
from ..cmake_variables import PYLOCO_LIB_PATH
from .PylocoEnv import PylocoEnv
from .VanillaEnv import VanillaEnv

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)


from pylocogym.data.deep_mimic_bob_adapter import (
    BobMotionDataFieldNames,
    DeepMimicMotionBobAdapter,
)
from pylocogym.data.deep_mimic_motion import DeepMimicMotion, DeepMimicMotionDataFieldNames
from pylocogym.data.lerp_dataset import LerpMotionDataset
from pylocogym.data.loop_dataset import LoopKeyframeMotionDataset
from pylocogym.envs.rewards.bob.humanoid_reward_task import TaskReward


class TaskEnv(VanillaEnv):
    def __init__(self, 
                 max_episode_steps, 
                 env_params, reward_params, 
                 enable_rand_init=True,
                 add_task_obs=True,
                 pretrained_model=None):
        
        self.add_task_obs = add_task_obs
        
        super().__init__(max_episode_steps, env_params, reward_params, enable_rand_init)
        
        self.reward_utils = TaskReward(
                self.cnt_timestep_size,
                self.num_joints,
                self.adapter.mimic_joints_index,
                reward_params,
                self.add_task_obs
            )
        
        if self.add_task_obs:
            self.heading_vector = np.array(reward_params["heading_vec"])
            self.forward_speed = reward_params["fwd_vel_cmd"]
            self.augment_obs_space()
            
        
    def augment_obs_space(self):
        self.observation_low = np.concatenate((
            self.observation_low,
            np.array([-1,-1]),  # heading direction
            np.float64(-10)     # forward velocity
        ),axis=None)

        self.observation_high = np.concatenate((
            self.observation_high,
            np.array([1,1]),  # heading direction
            np.float64(10)     # forward velocity
        ),axis=None)
        
        self.observation_space = spaces.Box(
            low=self.observation_low,
            high=self.observation_high,
            shape=(len(self.observation_low),),
            dtype=np.float64)   

    def get_obs(self):
        obs =  super().get_obs()
        obs = np.concatenate((obs,self.heading_vector,self.forward_speed), axis=None)
        return obs

