import importlib.util
import sys
from typing import List

import numpy as np
from gym import spaces
from ..cmake_variables import PYLOCO_LIB_PATH
from .VanillaEnv import VanillaEnv
from scipy.spatial.transform import Rotation

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)

from pylocogym.envs.rewards.bob.humanoid_reward_task import TaskReward


class TaskEnv(VanillaEnv):
    """
    The task that the agent is obligated to follow in the current implementation of the environment is a heading direction.
    """
    def __init__(self, 
                 max_episode_steps, 
                 env_params, reward_params, 
                 enable_rand_init=True,
                 add_task_obs=True,
                 pretrained_model=None):
        
        self.add_task_obs = add_task_obs
        self.heading_angle = 0
        self.angle_pace = 0.01
        self.heading_vector = np.array([0,1])
        
        super().__init__(max_episode_steps, env_params, reward_params, enable_rand_init)
        
        self.reward_utils = TaskReward(
                self.cnt_timestep_size,
                self.num_joints,
                self.adapter.mimic_joints_index,
                reward_params,
                self.add_task_obs
            )
        
        if self.add_task_obs:
            self.augment_obs_space()
            
        
    def augment_obs_space(self):
        """
        Augments (appends) the observation space to include the newly introduced  heading direction.
        """
        self.observation_low = np.concatenate((
            self.observation_low,
            np.array([-1,-1]),  # heading direction
            np.float64(-np.pi)    # heading angle
        ),axis=None)

        self.observation_high = np.concatenate((
            self.observation_high,
            np.array([1,1]),  # heading direction
            np.float64(np.pi)     # heading angle
        ),axis=None)
        
        self.observation_space = spaces.Box(
            low=self.observation_low,
            high=self.observation_high,
            shape=(len(self.observation_low),),
            dtype=np.float64)   

    def get_obs(self):
        """
        Augments (appends) the observation to include the newly introduced  heading direction.
        """
        obs =  super().get_obs()
        obs = np.concatenate((obs,self.heading_vector,self.heading_angle), axis=None)
        return obs

    def rotate_coordinate(self, q, qdot, angle, enable_pos_rotate = True):
        
        # Extract rotation matrix of the angle
        R_heading = Rotation.from_rotvec(angle * np.array([0, 1, 0]))
        
        # The "forward" direction is the z-axis.
        self.heading_vector = R_heading.apply(np.array([0,0,1]))
        self.heading_vector = self.heading_vector[[0,2]] # Remove the y-coordinate of the velocity, as the agent cannot fly.
        R_now = Rotation.from_euler("YXZ",q[3:6])
        R = R_heading*R_now # Compose the rotation R_now -> R_heading.
        
        if enable_pos_rotate:
            q[0:3] =  R_heading.apply(q[0:3]) # Apply heading rotation on the full motion clip. Intuition is easier if reminded that q is w.r.t world coordinates.
        q[3:6] = R.as_euler("YXZ") # Apply the full rotation on the *yaw, pitch & roll* of the agent. Essentially, this orients the agent.
        qdot[0:3] = R_heading.apply(qdot[0:3]) # Make the linear velocity vector "turn" to align with the "self.heading_angle".
        #not calculating the qdot angular velocity values, need to change order of axes before doing so
        # qdot[3:6] = R_heading.apply(qdot[3:6])

        return q, qdot

    def get_initial_state(self, initial_time):
        # Get desired root state, joint state according to phase
        now_t = initial_time

        # Load retargeted data
        res = self.lerp.eval(now_t)
        assert res is not None
        sample, kf = res
        sample_retarget = self.adapter.adapt(sample, kf)  # type: ignore # data after retargeting
        
        q_reset = sample_retarget.q
        qdot_reset = sample_retarget.qdot
        qdot_reset = qdot_reset * self.clips_play_speed

        return self.rotate_coordinate(q_reset, qdot_reset, self.heading_angle)
    
    def reset(self, seed=None, return_info=False, options=None, phase=0):
        # super().reset(seed=seed)  # We need this line to seed self.np_random
        self.current_step = 0
        self.box_throwing_counter = 0
        self.lerp.reset()  # reset dataloader

        # At the beginning of each episode, an initial heading angle in the semicircle is sampled.
        # The model will attempt to follow this direction for the whole duration of the episode,
        # The (uniform) sampling is done, so as to explore the solution space and make the model generalizable and robust.
        # self.heading_angle = np.random.uniform(-np.pi/2, np.pi/2)
        self.heading_angle = 0
        # self.phase = self.sample_initial_state()
        # Have the episode run starting from a random frame of the motion clip.
        # This is so that the agent adapts their parameterization (e.g. NN weights) in such a way that they can handle the motion holistically.
        # As a counter argument, if the agent were to always start from a single, predetermined phase (e.g. at the beginning of them motion),
        # then the agent would first learn to tackle the first few frames starting from the predetermined phase,
        # but then having to significantly tweak its parameterization to handle the full motion.
        # In a nutshell, this is done in the hopes that the agent converges faster to the desired solution.
        if self.enable_rand_init:
            self.initial_time = np.random.uniform(0, self.motion.duration-self.cnt_timestep_size)
            self.phase = self.initial_time / self.motion.duration
        else:
            self.phase = phase
            self.initial_time = self.phase * self.motion.duration # data scale

        total_duration = self.clips_repeat_num * self.motion.duration # data
        data_duration = total_duration - self.initial_time
        sim_duration = data_duration / self.clips_play_speed


        # Maximum episode step
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
        if self.heading_angle >= np.pi * 0.25:
            self.angle_pace = 0
        else:
            self.angle_pace = 0.01
        self.heading_angle += self.angle_pace
        self.steering_control()
        observation = self.get_obs()

        # update variables
        self.current_step += 1
        # self.action_buffer = np.roll(self.action_buffer, self.num_joints)  # moving action buffer
        # self.action_buffer[0 : self.num_joints] = action_applied

        # Accelerate or decelerate motion clips, usually deceleration
        # (clips_play_speed < 1 nomarlly)
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

        (sample_retarget.q, sample_retarget.qdot) = self.rotate_coordinate(sample_retarget.q, sample_retarget.qdot, self.heading_angle)
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
    
    def steering_control(self):
        temp_q = self._sim.get_q()
        temp_qdot = self._sim.get_qdot()
        (q_steer, qdot_steer) = self.rotate_coordinate(temp_q,temp_qdot,self.angle_pace,enable_pos_rotate=False)
        self._sim.set_q_and_qdot(q_steer,qdot_steer)
