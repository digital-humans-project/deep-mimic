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
        com_reward = weight_com * np.exp(-diff**2/(2.0*sigma_com**2))
        com_err = diff

        # Root height reward
        height = observation.y
        desired_height = sample_retarget.q_fields.root_pos[1]
        diff_squere = (height - desired_height)**2
        sigma_height = params.get("sigma_height", 0)
        weight_height = params.get("weight_height", 0)
        height_reward = weight_height * np.exp(-diff_squere/(2.0*sigma_height**2))
        height_err = height - desired_height

        # Root orientation reward
        R = Rotation.from_euler('YXZ',sample_retarget.q_fields.root_rot)
        desired_ori = R.as_quat()
        diff_squere = np.sum((observation.ori_q - desired_ori)**2)
        weight_root_ori = params.get("weight_root_ori", 0)
        sigma_root_ori = params.get("sigma_root_ori", 0)
        root_ori_reward = weight_root_ori * np.exp(-diff_squere/(sigma_root_ori**2))
        root_ori_err = diff_squere

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
        # ignore x axis diff
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

        # Task reward
        
        if self.add_task_obs:
            desired_v_xz = observation.observation[-1]
            desired_direction = observation.observation[-3:-1]
        else:
            desired_v_xz = params.get("fwd_vel_cmd",0)
            desired_direction = np.array(params.get("heading_vec",0))
            
        now_v_xz = np.sum(observation.vel[[0,2]]*desired_direction)/np.linalg.norm(desired_direction)
        diff = np.max([0.0,desired_v_xz - now_v_xz])
        weight_task = params.get("weight_task", 0)
        sigma_task = params.get("sigma_task", 0)
        task_reward = weight_task * np.exp(-diff/(2.0*sigma_task**2))
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