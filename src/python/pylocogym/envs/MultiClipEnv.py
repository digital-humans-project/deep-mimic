import importlib.util
import sys
import os
from importlib.machinery import SourceFileLoader
from typing import List
from src.python.pylocogym.utils import find_closest_frames

import numpy as np

from ..cmake_variables import PYLOCO_LIB_PATH
from .PylocoEnv import PylocoEnv

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)


from src.python.pylocogym.data.deep_mimic_bob_adapter import (
    BobMotionDataFieldNames,
    DeepMimicMotionBobAdapter,
)
from src.python.pylocogym.data.deep_mimic_motion import DeepMimicMotion, DeepMimicMotionDataFieldNames
from src.python.pylocogym.data.deep_mimic_combine_data import DeepMimicMotionCombine
from src.python.pylocogym.data.lerp_dataset import LerpMotionDataset
from src.python.pylocogym.data.loop_dataset import LoopKeyframeMotionDataset
from src.python.pylocogym.envs.rewards.bob.humanoid_reward import Reward


class MultiClipEnv(PylocoEnv):
    def __init__(self, max_episode_steps, env_params, reward_params, enable_rand_init=True):
        sim_dt = 1.0 / env_params["simulation_rate"]
        con_dt = 1.0 / env_params["control_rate"]

        # if env_params['robot_model'] == "Dog":
        #     robot_id = 0
        # elif env_params['robot_model'] == "Go1":
        #     robot_id = 1
        # elif env_params['robot_model'] == "Bob":
        robot_id = 2

        loadVisuals = False
        super().__init__(pyloco.VanillaSimulator(sim_dt, con_dt, robot_id, loadVisuals), env_params, max_episode_steps)

        self.enable_rand_init = enable_rand_init

        self._sim.lock_selected_joints = env_params.get("lock_selected_joints", False)
        self.enable_box_throwing = env_params.get("enable_box_throwing", False)
        self.box_throwing_interval = 100
        self.box_throwing_strength = 2
        self.box_throwing_counter = 0

        self.cnt_timestep_size = self._sim.control_timestep_size  # this should be equal to con_dt
        self.current_step = 0

        self.reward_params = reward_params
        self.sum_episode_reward_terms = {}
        self.sum_episode_err_terms = {}

        # self.action_buffer = np.zeros(self.num_joints * 3)  # history of actions [current, previous, past previous]

        self.rng = np.random.default_rng(env_params.get("seed", 1))  # create a local random number generator with seed

        # Set maximum episode length according to motion clips
        self.clips_play_speed = reward_params["clips_play_speed"]  # play speed for motion clips
        self.clips_repeat_num = reward_params["clips_repeat_num"]  # the number of times the clip needs to be repeated

        self.initial_pose = np.concatenate(
            [
                np.array([0, self._sim.nominal_base_height, 0, 0, 0, 0]),
                self.joint_angle_default,
            ]
        )

        # Dataloader 

        #accessing the motion clip from the json params
        self.all_motion_clips = reward_params["motion_clips_file_path"]
        self.num_motion_clips = len(self.all_motion_clips)
        self.frame_transition_idx = []
        #looping through the current and next motion file in the list
        for i in range(self.num_motion_clips-1):
            
            #loading first motion and creating its dataset and adapter
            motion_curr_path = os.path.join("data", "deepmimic", "motions", self.all_motion_clips[i])
            motion_curr_data = DeepMimicMotion(motion_curr_path)
            motion_curr_adapter = DeepMimicMotionBobAdapter(self.all_motion_clips[i], 
                                                            self.num_joints, 
                                                            self.joint_angle_limit_low, 
                                                            self.joint_angle_limit_high, 
                                                            self.joint_angle_default,)
            
            #loading second motion and creating its dataset and adapter
            motion_next_path = os.path.join("data", "deepmimic", "motions", self.all_motion_clips[i+1])
            motion_next_data = DeepMimicMotion(motion_next_path)
            motion_next_adapter = DeepMimicMotionBobAdapter(self.all_motion_clips[i+1], 
                                                            self.num_joints, 
                                                            self.joint_angle_limit_low, 
                                                            self.joint_angle_limit_high, 
                                                            self.joint_angle_default,)
            
            #comparing the euler joint angles of all the frames in both the motion to find the 
            #index values at which motions should transition
            self.frame_transition_idx.append(find_closest_frames(
                motion_curr_adapter.retarget_datamotion(motion_curr_data.q),
                motion_next_adapter.retarget_datamotion(motion_next_data.q)
                ))

        #creating the final motion dataset with transitions incorporated    
        self.motion = DeepMimicMotionCombine(self.all_motion_clips, 
                                             self.frame_transition_idx, 
                                             self.clips_repeat_num)
        
        #looping is already taken care of in DeepMimicMotionCombine so the num_loop=1
        self.loop = LoopKeyframeMotionDataset(
            self.motion, num_loop=1, track_fields=[BobMotionDataFieldNames.ROOT_POS]
        )

        #doing all the required interpolations
        self.lerp = LerpMotionDataset(
            self.loop,
            lerp_fields=[
                DeepMimicMotionDataFieldNames.ROOT_POS,
            ],
            alerp_fields=[
                DeepMimicMotionDataFieldNames.R_KNEE_ROT,
                DeepMimicMotionDataFieldNames.L_KNEE_ROT,
                DeepMimicMotionDataFieldNames.R_ELBOW_ROT,
                DeepMimicMotionDataFieldNames.L_ELBOW_ROT,
            ],
            slerp_fields=[
                DeepMimicMotionDataFieldNames.ROOT_ROT,
                DeepMimicMotionDataFieldNames.CHEST_ROT,
                DeepMimicMotionDataFieldNames.NECK_ROT,
                DeepMimicMotionDataFieldNames.R_HIP_ROT,
                DeepMimicMotionDataFieldNames.R_ANKLE_ROT,
                DeepMimicMotionDataFieldNames.R_SHOULDER_ROT,
                DeepMimicMotionDataFieldNames.L_HIP_ROT,
                DeepMimicMotionDataFieldNames.L_ANKLE_ROT,
                DeepMimicMotionDataFieldNames.L_SHOULDER_ROT,
            ],
        )

        #adapting DeepMimic data to Bob suitable format
        self.adapter = DeepMimicMotionBobAdapter(
            reward_params["motion_clips_file_path"],
            self.num_joints,
            self.joint_angle_limit_low,
            self.joint_angle_limit_high,
            self.joint_angle_default,
            # initial_pose=self.initial_pose,
        )


        # Reward class
        self.reward_utils = Reward(
            self.cnt_timestep_size,
            self.num_joints,
            self.adapter.mimic_joints_index,
            reward_params,
        )

        # Forwards Kinematics class
        # self.fk = ForwardKinematics(env_params["urdf_path"])
        self.minimum_height = 0.009

    def reset(self, seed=None, return_info=False, options=None, phase=0):
        # super().reset(seed=seed)  # We need this line to seed self.np_random
        self.current_step = 0
        self.box_throwing_counter = 0
        self.lerp.reset()  # reset dataloader

        # self.phase = self.sample_initial_state()
        if self.enable_rand_init:
            self.initial_time = np.random.uniform(0, self.motion.duration-self.cnt_timestep_size)
            self.phase = self.initial_time / self.motion.duration
        else:
            self.phase = phase
            self.initial_time = self.phase * self.motion.duration # data scale

        #total_duration = self.clips_repeat_num * self.motion.duration # data
        total_duration = self.motion.duration
        data_duration = total_duration - self.initial_time
        sim_duration = data_duration / self.clips_play_speed


        # Maximum episdode step
        self.max_episode_steps = int(sim_duration / self.cnt_timestep_size)
        assert self.max_episode_steps > 0, "max_episode_steps should be positive"

        (q_reset, qdot_reset) = self.get_initial_state(self.initial_time)
        self._sim.reset(q_reset, qdot_reset, self.initial_time / self.clips_play_speed)  # q, qdot include root's state(pos,ori,vel,angular vel)
        # self._sim.reset()

        observation = self.get_obs()
        self.sum_episode_reward_terms = {}
        self.sum_episode_err_terms = {}
        # self.action_buffer = np.concatenate(
        #     (self.joint_angle_default, self.joint_angle_default, self.joint_angle_default), axis=None
        # )

        info = {"msg": "===Episode Reset Done!===\n"}
        return (observation, info) if return_info else observation

    def step(self, action: List[np.ndarray]):
        # throw box if needed
        if self.enable_box_throwing and self.current_step % self.box_throwing_interval == 0:
            random_start_pos = (self.rng.random(3) * 2 - np.ones(3)) * 2  # 3 random numbers btw -2 and 2
            self._sim.throw_box(self.box_throwing_counter % 3, self.box_throwing_strength, random_start_pos)
            self.box_throwing_counter += 1

        # run simulation
        action_applied = self.scale_action(action)
        self._sim.step(action_applied)
        observation = self.get_obs()

        # update variables
        self.current_step += 1
        now_t = self._sim.get_time_stamp() * self.clips_play_speed

        """ Forwards and Inverse kinematics """
        # Load retargeted data
        res = self.lerp.eval(now_t)
        assert res is not None, "lerp.eval(now_t) is None"
        sample, kf = res
        sample_retarget = self.adapter.adapt(sample, kf)  # type: ignore # data after retargeting

        # data_joints = sample_retarget.q
        # q_desired = self._sim.get_ik_solver_q(data_joints,
        #                                       end_effectors_pos[0,:],
        #                                       end_effectors_pos[1,:],
        #                                       end_effectors_pos[2,:],
        #                                       end_effectors_pos[3,:])

        end_effectors_raw = self._sim.get_fk_ee_pos(sample_retarget.q)
        end_effectors_pos = np.array(
            [end_effectors_raw[0], end_effectors_raw[2], end_effectors_raw[1], end_effectors_raw[3]]
        )
        for each_pos in end_effectors_pos:
            if each_pos[1] < self.minimum_height:
                each_pos[1] = self.minimum_height
        # sample_retarget.q = q_desired

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
        terminated, truncated, term_info = self.is_done(observation)
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

    def filter_actions(self, action_new, action_old, max_joint_vel):
        # with this filter we have much fewer cases that joints cross their limits, but still not zero
        threshold = max_joint_vel * np.ones(self.num_joints)  # max angular joint velocity
        diff = action_new - action_old
        action_filtered = action_old + np.sign(diff) * np.minimum(np.abs(diff), threshold)
        return action_filtered
