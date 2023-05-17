import pathlib
import datetime
import time
from pylocogym.data.deep_mimic_motion import DeepMimicMotion
from pylocogym.data.lerp_dataset import LerpMotionDataset
from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from scipy.spatial.transform import Rotation as R


# ? (Konstantinos)
# Why do we have the "Retarget" class in both this moduel ("retarget.py") and inside the "VanillaEnv.py"?
# I believe it would make more sense to have it only inside the current module and import it wherever needed.

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
            # self.joint_angle_default = np.zeros(44) # Suggestion for abbreviating the above array definition. 
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



def test(params, reward_path=None, data_path = None):
    """Render environment using given action"""

    # =============
    # unpack params
    # =============

    env_id = params['env_id']
    env_params = params['environment_params']
    hyp_params = params['train_hyp_params']
    reward_params = params['reward_params']

    max_episode_steps = hyp_params.get('max_episode_steps', 5000)
    seed = hyp_params.get("seed", 313)
    env_kwargs = {"max_episode_steps": max_episode_steps, "env_params": env_params, "reward_params": reward_params}

    if reward_path is not None:
        reward_params["reward_file_path"] = reward_path  # add reward path to reward params

    # =============
    # load motion clips data
    # =============

    if data_path is not None:
        dataset = DeepMimicMotion(data_path) # Do not time shift the dataset
        lerp = LerpMotionDataset(dataset)
        # ? (Konstantinos) I'm having trouble understanding the flow of data in the pipeline and what each class represents :/

    # =============
    # create a single environment for evaluation
    # =============

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )

    # =============
    # joint config 
    # =============

    # ? (Konstantinos) Where were the specifications of the joints lower and upper limits  provided?
    joint_lower_limit = np.array([-0.5 , -1.2 , -1.2 , -0.5 , -0.5 , -1.2 , -0.5 , -0.75, -0.75,
                                -0.5 ,  0.  ,  0.  , -0.5 , -0.5 , -0.5 , -0.5 , -0.5 , -0.5 ,
                                -0.5 , -0.8 , -0.2 , -0.75, -0.75, -0.5 , -0.2 , -0.6 , -0.5 ,
                                -3.  , -3.  , -0.5 , -0.3 , -1.2 , -0.5 , -0.75, -0.5 , -0.5 ,
                                -2.5 , -2.5 , -1.  , -1.6 , -0.5 , -0.5 , -1.  , -1.  ])
    joint_upper_limit = np.array([0.5 , 1.2 , 1.2 , 0.5 , 1.2 , 0.5 , 0.5 , 0.75, 0.75, 0.5 , 2.,
                                2.  , 0.5 , 0.9 , 0.9 , 0.5 , 0.5 , 0.5 , 0.5 , 0.2 , 0.8 , 0.2 ,
                                0.2 , 0.5 , 0.6 , 0.2 , 0.5 , 0.5 , 0.5 , 0.5 , 1.2 , 0.3 , 0.5 ,
                                1.  , 0.75, 0.5 , 0.  , 0.  , 1.6 , 1.  , 0.5 , 0.5 , 1.  , 1.  ])
    
    
    # =============
    # start playing
    # =============
    episodes = 100
    # Remember the frame rate provided below is independent of the frame rate at which the data was recorded.
    # That is possible thanks to the linear interpolation done between data frames. (see "lerp_dataset.py -> eval()").
    # NOTE: IDEALLY, we should have the frame rate be identical to the one provided in the data, so as to avoid unexpected artefacts.
    frame_rate = 60 
    action_shape = eval_env.action_space.shape[0] 
    retarget=Retarget(action_shape,joint_lower_limit,joint_upper_limit)

    for ep in range(episodes):
        eval_env.reset() # Again, this is copied from the 
        done = False
        t = 0
        while not done:
            eval_env.render("human")
            action = retarget.retarget_joint_angle(lerp.eval(t).q)
            obs, reward, done, info = eval_env.step([action])
            t += 1.0/frame_rate
            time.sleep(0.1) # ? (Konstantinos) Why is the sleeping required? 
       
            
    eval_env.close()
