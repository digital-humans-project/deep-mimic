"""
Computing reward for Vanilla setup, constant target speed, gaussian kernels
"""
import numpy as np
from pylocogym.envs.rewards.utils.utils import *
from pylocogym.envs.rewards.bob.humanoid_retarget import Retarget

class Reward:
    def __init__(self, cnt_timestep, num_joints, joint_lower_limit, joint_higher_limit, reward_params):
        """
        :variable dt: control time step size
        :variable num_joints: number of joints
        :variable params: reward params read from the config file
        """
        self.dt = cnt_timestep
        self.num_joints = num_joints
        self.params = reward_params
        self.retarget = Retarget(num_joints,joint_lower_limit,joint_higher_limit)
        

    def compute_reward(self, observation_raw, all_torques, action_buffer, is_obs_fullstate,
                nominal_base_height, dataloader):
        """
        Compute the reward based on observation (Vanilla Environment).

        :param observation_raw: current observation
        :param all_torques: torque records during the last control timestep
        :param action_buffer: history of previous actions
        :param is_obs_fullstate: flag to choose full state obs or not.
        
        :return: total reward, reward information (different terms can be passed here to be plotted in the graphs)
        """

        # test_data = {'observation_raw': observation_raw, 'dt': dt, 'num_joints': num_joints, 'params': params,
        #              'feet_status': feet_status, 'all_torques': all_torques, 'action_buffer': action_buffer,
        #              'is_obs_fullstate': is_obs_fullstate, 'joint_angles_default': joint_angles_default,
        #              'nominal_base_height': nominal_base_height}
        
        num_joints = self.num_joints
        dt = self.dt
        params = self.params

        observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)

        action_dot, action_ddot = calc_derivatives(action_buffer, dt, num_joints)
        cmd_fwd_vel = params.get("fwd_vel_cmd", 1.0)
        torque = tail(all_torques, num_joints)

        now_t = observation.time_stamp
        
        # =============
        # define cost/reward terms here:
        # =============

        # x&z position corresponds
        # MOTION CLIPS : X FORWARD, Z RIGHT, Y UP
        # OURS MODEL: Z FORWARD, X LEFT, Y UP

        if dataloader.eval(now_t) is None:
            print(now_t)
        desired_base_pos_x = dataloader.eval(now_t).q[0]/dataloader.eval(now_t).q[1]
        now_base_z = observation.pos[2]/observation.pos[1]
        diff = np.linalg.norm(desired_base_pos_x - now_base_z)
        sigma_vel = params.get("sigma_velocity", 0)
        weight_vel = params.get("weight_velocity", 0)
        forward_vel_reward = weight_vel * np.exp(-diff**2/(2.0*sigma_vel**2))

        height = observation.y
        diff_squere = (height - nominal_base_height)**2
        sigma_height = params.get("sigma_height", 0)
        weight_height = params.get("weight_height", 0)
        height_reward = weight_height * np.exp(-diff_squere/(2.0*sigma_height**2))

        roll_squere = (observation.roll)**2
        pitch_squere = (observation.pitch)**2
        weight_attitude = params.get("weight_attitude", 0)
        sigma_attitude = params.get("sigma_attitude", 0)
        attitude_reward = weight_attitude * np.exp(-(roll_squere + pitch_squere)/(4.0*sigma_attitude**2))

        N = num_joints
        weight_torque = params.get("weight_torque", 0)
        sigma_torque = params.get("sigma_torque", 0)
        torque_reward = weight_torque * np.exp(-np.sum(torque**2)/(2.0*N*sigma_torque**2))

        weight_smoothness1 = params.get("weight_smoothness1", 0)
        sigma_smoothness1 = params.get("sigma_smoothness1", 0)
        smoothness1_reward = weight_smoothness1 * np.exp(-np.sum(action_dot**2)/(2.0*N*sigma_smoothness1**2))

        weight_smoothness2 = params.get("weight_smoothness2", 0)
        sigma_smoothness2 = params.get("sigma_smoothness2", 0)    
        smoothness2_reward = weight_smoothness2 * np.exp(-np.sum(action_ddot**2)/(2.0*N*sigma_smoothness2**2))

        # Motion imitation reward #
        joint_angles = observation.joint_angles
        desired_angles = self.retarget.retarget_joint_angle(dataloader.eval(now_t).q)
        diff = joint_angles - desired_angles
        weight_joints = params.get("weight_joints", 0)
        sigma_joints = params.get("sigma_joints", 0)
        joint_reward = weight_joints * np.exp(-np.sum(np.square(diff))/(2.0*N*sigma_joints**2))

        # =============
        # sum up rewards
        # =============
        smoothness_reward = params.get("weight_smoothness", 0) * (smoothness1_reward + smoothness2_reward)
        reward = forward_vel_reward + smoothness_reward + torque_reward + height_reward + attitude_reward + joint_reward

        info = {
            "forward_vel_reward": forward_vel_reward,
            "height_reward": height_reward,
            "attitude_reward": attitude_reward,

            "torque_reward": torque_reward,
            "smoothness1_reward": smoothness1_reward,
            "smoothness2_reward": smoothness2_reward,
            "smoothness_reward": smoothness_reward,

            "joint_reward": joint_reward
        }

        return reward, info


    def punishment(self, current_step, max_episode_steps):  # punishment for early termination
        penalty = self.params['weight_early_penalty'] * (max_episode_steps - current_step)
        return penalty
