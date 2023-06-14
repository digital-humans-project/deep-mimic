"""
Computing reward for Vanilla setup, constant target speed, gaussian kernels
"""
import numpy as np
from pylocogym.envs.rewards.utils.utils import *
from scipy.spatial.transform import Rotation
from .humanoid_reward import Reward

class TaskReward(Reward):
    def __init__(self, 
                 cnt_timestep,
                 num_joints, 
                 mimic_joints_index, 
                 reward_params,
                 add_task_obs):
        super().__init__(cnt_timestep, num_joints, mimic_joints_index, reward_params)
        self.add_task_obs = add_task_obs


    def compute_reward(self, 
                       observation_raw, 
                       is_obs_fullstate,
                       sample_retarget,
                       end_effectors_pos):
        """
        Compute the reward based on observation (Vanilla Environment).

        :param observation_raw: current observation
        :param all_torques: torque records during the last control timestep
        :param action_buffer: history of previous actions
        :param is_obs_fullstate: flag to choose full state obs or not.
        
        :return: total reward, reward information (different terms can be passed here to be plotted in the graphs)
        """
        
        num_joints = self.num_joints
        dt = self.dt
        params = self.params

        observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)
        
        # =======================
        # OURS MODEL coordinate   : Z FORWARD, X LEFT, Y UP
        # =======================

        # Root position reward
        desired_base_pos_xz = sample_retarget.q_fields.root_pos
        now_base_xz = observation.pos
        diff = np.linalg.norm(desired_base_pos_xz - now_base_xz)
        sigma_com = params.get("sigma_com", 0)
        weight_com = params.get("weight_com", 0)
        com_reward = weight_com * np.exp(-diff**2/(2.0*sigma_com**2)) # Inspired from assignment 2 equivalent reward.
        com_err = diff # Error which will be logged.

        # Root height reward
        height = observation.y
        desired_height = sample_retarget.q_fields.root_pos[1]
        diff_square = (height - desired_height)**2
        sigma_height = params.get("sigma_height", 0)
        weight_height = params.get("weight_height", 0)
        height_reward = weight_height * np.exp(-diff_square/(2.0*sigma_height**2)) # Inspired from assignment 2 equivalent reward.
        height_err = height - desired_height # Error which will be logged.

        # Root orientation reward
        R = Rotation.from_euler('YXZ',sample_retarget.q_fields.root_rot)
        desired_ori = R.as_quat()
        diff_square = np.sum((observation.ori_q - desired_ori)**2)
        weight_root_ori = params.get("weight_root_ori", 0)
        sigma_root_ori = params.get("sigma_root_ori", 0)
        root_ori_reward = weight_root_ori * np.exp(-diff_square/(sigma_root_ori**2))
        root_ori_err = diff_square

        N = num_joints
        N_mimic = len(self.mimic_joints_index)
        # set_other = set(range(N)) - self.mimic_joints_index # joints that have no corresponding in motion clips


        motion_joints = sample_retarget.q_fields.joints
        motion_joints_dot = sample_retarget.qdot_fields.joints
        # Motion imitation reward 
        joint_angles = observation.joint_angles[list(self.mimic_joints_index)]
        desired_angles = motion_joints[list(self.mimic_joints_index)]
        diff = np.sum((joint_angles - desired_angles)**2)
        weight_joints = params.get("weight_joints", 0)
        sigma_joints = params.get("sigma_joints", 0)
        joints_reward = weight_joints * np.exp(-diff/(2.0*N_mimic*sigma_joints**2))
        joints_err = diff


        # End effector reward
        diff_lf = end_effectors_pos[0] - observation.lf
        diff_rf = end_effectors_pos[1] - observation.rf
        diff_lh = end_effectors_pos[2] - observation.lh
        diff_rh = end_effectors_pos[3] - observation.rh
        sum_diff_square = np.sum(np.square(diff_lf)) + np.sum(np.square(diff_rf)) \
                        + np.sum(np.square(diff_lh)) + np.sum(np.square(diff_rh))
 
        weight_end_effectors = params.get("weight_joints_vel", 0)
        sigma_end_effectors = params.get("sigma_joints_vel", 0)
        end_effectors_reward = weight_end_effectors * np.exp(-sum_diff_square/(2.0*4*sigma_end_effectors**2))
        end_effectors_err = sum_diff_square

        # Joint velocities reward
        joint_velocities = observation.joint_vel[list(self.mimic_joints_index)]
        desired_velocities = motion_joints_dot[list(self.mimic_joints_index)]
        diff = np.sum((joint_velocities - desired_velocities)**2)
        weight_joints_vel = params.get("weight_joints_vel", 0)
        sigma_joints_vel = params.get("sigma_joints_vel", 0)
        joints_vel_reward = weight_joints_vel * np.exp(-diff/(2.0*N_mimic*sigma_joints_vel**2))
        joints_vel_err = diff

        # Task (follow specific direction) reward
        desired_heading_vector = observation.observation[-3:-1]
        now_heading_vector = observation.vel[[0,2]]
        
        # "diff" tracks the similarity of the direction the agent is heading towards in the current observation, "now_heading_vector",
        # and the desired direction, "desired_heading_vector".
        # Remember: # Math: \text{cosine similarity}(\vec{a}, \vec{b}) := \frac{\vec{a} \cdot \vec{b}}{ \| \vec{a} \| \| \vec{b} \|}
        # NOTE: 1) If the directions are known to be unit vectors (i.e. norm = 1), then there is no need to call "np.linalg.norm"
        # This can be computationally less expensive, because computing square roots (definition of a norm) requires an iterative root-problem solver.
        # 2) The vector "desired_heading_vector" is a unit vector by definition.
        diff = np.dot(now_heading_vector, desired_heading_vector)/np.linalg.norm(now_heading_vector)
        weight_task = params.get("weight_task", 0)
        sigma_task = params.get("sigma_task", 0)
        task_reward = weight_task * np.exp(-(diff-1)**2/(2.0*sigma_task**2))
        task_err = diff


        # =============
        # sum up rewards
        # =============
        smoothness1_reward = 0
        smoothness2_reward = 0
        smoothness_reward = params.get("weight_smoothness", 0) * (smoothness1_reward + smoothness2_reward)
        reward = com_reward \
                + smoothness_reward \
                + height_reward     \
                + root_ori_reward   \
                + joints_reward     \
                + joints_vel_reward \
                + end_effectors_reward \
                + task_reward

        info = {
            "com_reward": com_reward,
            "height_reward": height_reward,
            "root_ori_reward": root_ori_reward,

            "smoothness1_reward": smoothness1_reward,
            "smoothness2_reward": smoothness2_reward,
            "smoothness_reward": smoothness_reward,

            "joints_reward": joints_reward,

            "joints_vel_reward": joints_vel_reward,

            # "leg_reward": leg_reward,

            "end_effectors_reward": end_effectors_reward,

            "task_reward": task_err
        }

        err = {
            "com_err": com_err,
            "height_err": height_err,
            "root_ori_err": root_ori_err,
            "joints_err": joints_err,
            "end_effectors_err": end_effectors_err,
            "joints_vel_err": joints_vel_err,
            "task_err": task_err
        }

        return reward, info, err