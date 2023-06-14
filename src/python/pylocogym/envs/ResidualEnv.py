import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from typing import List

import numpy as np

from ..cmake_variables import PYLOCO_LIB_PATH
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
from pylocogym.data.deep_mimic_motion import (
    DeepMimicMotion,
    DeepMimicMotionDataFieldNames,
)
from pylocogym.data.forward_kinematics import ForwardKinematics
from pylocogym.data.lerp_dataset import LerpMotionDataset
from pylocogym.data.loop_dataset import LoopKeyframeMotionDataset
from pylocogym.envs.rewards.bob.humanoid_reward import Reward


class ResidualEnv(VanillaEnv):
    def __init__(self, 
                 max_episode_steps, 
                 env_params, reward_params, 
                 enable_rand_init=True):
        
        super().__init__(max_episode_steps, env_params, reward_params, enable_rand_init)
        
    def step(self, action: List[np.ndarray]):
        # throw box if needed
        if self.enable_box_throwing and self.current_step % self.box_throwing_interval == 0:
            random_start_pos = (self.rng.random(3) * 2 - np.ones(3)) * 2  # 3 random numbers btw -2 and 2
            self._sim.throw_box(self.box_throwing_counter % 3, self.box_throwing_strength, random_start_pos)
            self.box_throwing_counter += 1

        # Accelerate or decelerate motion clips, usually deceleration
        # (clips_play_speed < 1 nomarlly)
        next_t = (self._sim.get_time_stamp() + self.cnt_timestep_size) * self.clips_play_speed

        """ Forwards and Inverse kinematics """
        # Load retargeted data
        res = self.lerp.eval(next_t)
        assert res is not None
        sample, kf = res
        sample_retarget = self.adapter.adapt(sample, kf)  # type: ignore # data after retargeting
        
        # run simulation
        target_act = action * np.pi + sample_retarget.q_fields.joints
        action_applied = np.clip(target_act, self.joint_angle_limit_low, self.joint_angle_limit_high)
        self._sim.step(action_applied)
        observation = self.get_obs()

        now_t = self._sim.get_time_stamp() * self.clips_play_speed
        assert abs(now_t - next_t) < 1e-6, f"{now_t} - {next_t} = {now_t - next_t}"

        # update variables
        self.current_step += 1
        
        end_effectors_raw = self._sim.get_fk_ee_pos(sample_retarget.q)
        end_effectors_pos = np.array(
            [end_effectors_raw[0], end_effectors_raw[2], end_effectors_raw[1], end_effectors_raw[3]]
        )

        if self.use_ik_solution: # fix the problematic ee_pos and joints' values
            (sample_retarget.q, end_effectors_pos)  = self.get_ik_solutions(sample_retarget.q, end_effectors_pos)
        
        else: # simply just fix the problematic ee_pos, not changed the joints' values
            for each_pos in end_effectors_pos:
                if each_pos[1] < self.minimum_height:
                    each_pos[1] = self.minimum_height

        # compute reward
        reward, reward_info, err_info = self.reward_utils.compute_reward(
            observation,
            self.is_obs_fullstate,
            sample_retarget,
            end_effectors_pos,
        )

        self.sum_episode_reward_terms = {
            key: self.sum_episode_reward_terms.get(key, 0) + reward_info.get(key, 0) for key in reward_info.keys()
        }

        self.sum_episode_err_terms = {
            key: self.sum_episode_err_terms.get(key, 0) + err_info.get(key, 0) for key in err_info.keys()
        }

        # check if episode is done
        terminated, truncated, term_info = self.is_done(observation, sample_retarget)
        done = terminated | truncated

        # punishment for early termination
        if terminated:
            reward -= self.reward_utils.punishment(self.current_step, self.max_episode_steps)

        info = {
            "is_success": truncated,
            "termination_info": term_info,
            "current_step": self.current_step,
            "action_applied": action_applied,
            "reward_info": reward_info,
            "TimeLimit.truncated": truncated,
            "msg": "=== 1 Episode Taken ===\n",
        }

        if done:
            mean_episode_reward_terms = {
                key: self.sum_episode_reward_terms.get(key, 0) / self.current_step for key in reward_info.keys()
            }
            mean_episode_err_terms = {
                key: self.sum_episode_err_terms.get(key, 0) / self.current_step for key in err_info.keys()
            }
            info["mean_episode_reward_terms"] = mean_episode_reward_terms
            info["mean_episode_err_terms"] = mean_episode_err_terms

        return observation, reward, done, info

