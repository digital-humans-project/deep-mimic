import sys
import numpy as np
import importlib.util
from importlib.machinery import SourceFileLoader

from .PylocoEnv import PylocoEnv
from ..cmake_variables import PYLOCO_LIB_PATH

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)


from scipy.spatial.transform import Rotation as R

class Retarget:
    def __init__(self, action_shape, joint_lower_limit, joint_upper_limit, joint_default_angle = None):

        self.joint_angle_limit_low = joint_lower_limit
        self.joint_angle_limit_high = joint_upper_limit
        self.joint_angle_default = joint_default_angle
        if self.joint_angle_default is None:
            self.joint_angle_default = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0 ])
        self.joint_scale_factors = np.maximum(abs(self.joint_angle_default - self.joint_angle_limit_low),
                                abs(self.joint_angle_default - self.joint_angle_limit_high))
        self.action_shape = action_shape
        
    def quart_to_rpy(self, q, mode):
        # q is in (w,x,y,z) format
        q_xyzw = list(q[1:])
        q_xyzw.append(q[0])
        r = R.from_quat(q_xyzw) 
        euler = r.as_euler(mode)
        return euler[0], euler[1], euler[2]
    
    def rescale_action(self, action):
        bound_action = np.minimum(np.maximum(action,self.joint_angle_limit_low),self.joint_angle_limit_high)
        scaled_action = (bound_action - self.joint_angle_default) / self.joint_scale_factors 
        return scaled_action
    
    def retarget_joint_angle(self, motion_clips_q, require_root = False):
        """Given a motion_clips orientation data, return a retarget action"""
        action = np.zeros(self.action_shape)
        if not require_root:

            (chest_z, chest_y, chest_x) = self.quart_to_rpy(motion_clips_q[7:11], 'zyx')
            (neck_z,  neck_y,  neck_x) = self.quart_to_rpy(motion_clips_q[11:15],'zyx')
            (r_hip_z, r_hip_x, r_hip_y) = self.quart_to_rpy(motion_clips_q[15:19],'zxy')
            (r_ankle_z, r_ankle_x, r_ankle_y) = self.quart_to_rpy(motion_clips_q[20:24],'zxy')
            (r_shoulder_z, r_shoulder_x, r_shoulder_y) = self.quart_to_rpy(motion_clips_q[24:28],'zxy')
            (l_hip_z, l_hip_x, l_hip_y) = self.quart_to_rpy(motion_clips_q[29:33],'zxy')
            (l_ankle_z, l_ankle_x, l_ankle_y) = self.quart_to_rpy(motion_clips_q[34:38],'zxy')
            (l_shoulder_z, l_shoulder_x, l_shoulder_y) = self.quart_to_rpy(motion_clips_q[38:42],'zxy')

            # chest - xyz euler angle 
            action[0] = -chest_z
            action[3] = chest_y
            action[6] = chest_x

            # neck - xyz euler angle 
            action[18] = -neck_z
            action[23] = neck_y
            action[26] = neck_x

            # shoulder - xzy euler angle 
            action[27] = -l_shoulder_z
            action[30] = l_shoulder_x
            action[33] = l_shoulder_y

            action[28] = -r_shoulder_z
            action[31] = r_shoulder_x
            action[34] = r_shoulder_y

            # ankle - xzy euler angle 
            action[13] = -l_ankle_z
            action[16] = l_ankle_x

            action[14] = -r_ankle_z
            action[17] = r_ankle_x            

            # hip - xzy euler angle 
            action[1] = -l_hip_z
            action[4] = l_hip_x
            action[7] = l_hip_y

            action[2] = -r_hip_z
            action[5] = r_hip_x
            action[8] = r_hip_y

            r_knee = motion_clips_q[19:20]
            r_elbow = motion_clips_q[28:29]
            l_knee = motion_clips_q[33:34]
            l_elbow = motion_clips_q[42:43]

            # elbow - revolute joint 
            action[36] = l_elbow
            action[37] = r_elbow

            # knee - revolute joint 
            action[10] = l_knee
            action[11] = r_knee

            action = self.rescale_action(action)

        return action

class VanillaEnv(PylocoEnv):

    def __init__(self, max_episode_steps, env_params, reward_params):
        sim_dt = 1.0 / env_params['simulation_rate']
        con_dt = 1.0 / env_params['control_rate']

        if env_params['robot_model'] == "Dog":
            robot_id = 0
        elif env_params['robot_model'] == "Go1":
            robot_id = 1
        elif env_params['robot_model'] == "Bob":
            robot_id = 2

        loadVisuals = False
        super().__init__(pyloco.VanillaSimulator(sim_dt, con_dt, robot_id, loadVisuals), env_params, max_episode_steps)

        self._sim.lock_selected_joints = env_params.get('lock_selected_joints', False)
        self.enable_box_throwing = env_params.get('enable_box_throwing', False)
        self.box_throwing_interval = 100
        self.box_throwing_strength = 2
        self.box_throwing_counter = 0

        if "reward_file_path" in reward_params.keys():
            reward_file_path = reward_params["reward_file_path"]

            self.reward_utils = SourceFileLoader('reward_utils', reward_file_path).load_module()
        else:
            raise Exception("Reward file not specified. Please specify via --rewardFile.")

        self.cnt_timestep_size = self._sim.control_timestep_size  # this should be equal to con_dt
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.reward_params = reward_params
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.zeros(self.num_joints * 3)  # history of actions [current, previous, past previous]

        self.rng = np.random.default_rng(
            env_params.get("seed", 1))  # create a local random number generator with seed

    def reset(self, seed=None, return_info=False, options=None):
        # super().reset(seed=seed)  # We need this line to seed self.np_random
        self.current_step = 0
        self.box_throwing_counter = 0
        self._sim.reset()
        observation = self.get_obs()
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.concatenate(
            (self.joint_angle_default, self.joint_angle_default, self.joint_angle_default), axis=None)

        info = {"msg": "===Episode Reset Done!===\n"}
        return (observation, info) if return_info else observation

    def step(self, action: [np.ndarray]):

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
        self.action_buffer = np.roll(self.action_buffer, self.num_joints)  # moving action buffer
        self.action_buffer[0:self.num_joints] = action_applied

        # compute reward
        reward, reward_info = self.reward_utils.compute_reward(observation, self.cnt_timestep_size, self.num_joints,
                                                               self.reward_params, self.get_feet_status(),
                                                               self._sim.get_all_motor_torques(), self.action_buffer,
                                                               self.is_obs_fullstate, self.joint_angle_default,
                                                               self._sim.nominal_base_height)

        self.sum_episode_reward_terms = {key: self.sum_episode_reward_terms.get(key, 0) + reward_info.get(key, 0) for
                                         key in reward_info.keys()}

        # check if episode is done
        terminated, truncated, term_info = self.is_done(observation)
        done = terminated | truncated

        # punishment for early termination
        if terminated:
            reward -= self.reward_utils.punishment(self.current_step, self.max_episode_steps, self.reward_params)

        info = {
            "is_success": truncated,
            "termination_info": term_info,
            "current_step": self.current_step,
            "action_applied": action_applied,
            "reward_info": reward_info,
            "TimeLimit.truncated": truncated,
            "msg": "=== 1 Episode Taken ===\n"
        }

        if done:
            mean_episode_reward_terms = {key: self.sum_episode_reward_terms.get(key, 0) / self.current_step for key in
                                         reward_info.keys()}
            info["mean_episode_reward_terms"] = mean_episode_reward_terms

        return observation, reward, done, info

    def filter_actions(self, action_new, action_old, max_joint_vel):
        # with this filter we have much fewer cases that joints cross their limits, but still not zero
        threshold = max_joint_vel * np.ones(self.num_joints)  # max angular joint velocity
        diff = action_new - action_old
        action_filtered = action_old + np.sign(diff) * np.minimum(np.abs(diff), threshold)
        return action_filtered
