from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os

def test(params, motion_clips_path=None, urdf_path = None):
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
    env_kwargs = {"max_episode_steps": max_episode_steps, "env_params": env_params, 
                  "reward_params": reward_params,"enable_rand_init": False}

    if motion_clips_path is not None:
        reward_params["motion_clips_file_path"] = motion_clips_path  # add reward path to reward params
    # if urdf_path is not None:
    #     env_params["urdf_path"] = urdf_path
        
    # =============
    # create a simple environment for evaluation
    # =============

    # eval_env = gym.make(env_id, **env_kwargs)

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv
    )
    
    # =============
    # start playing
    # =============
    episodes = 10000
    frame_rate = 60
    
    for ep in range(episodes):
        eval_env.reset()
        done = False
        t = eval_env.envs[0].phase*eval_env.envs[0].motion.duration
        # t2 = eval_env.envs[1].phase*eval_env.envs[1].dataset.duration
        while not done:
            eval_env.envs[0].render("human")
            sample, kf = eval_env.envs[0].lerp.eval(t)
            action = eval_env.envs[0].adapter.adapt(sample, kf).q[6:]
            # action = eval_env.envs[0].lerp.eval(t).q[6:]
            obs, reward, done, info = eval_env.envs[0].step(action)
            # print("now time, now phase", obs[-2],obs[-1])
            t += 1.0/(frame_rate/eval_env.envs[0].clips_play_speed)
            time.sleep(0.01)
    eval_env.close()

def play_motion(params, motion_clips_path=None, urdf_path = None):
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
    env_kwargs = {"max_episode_steps": max_episode_steps, "env_params": env_params, 
                  "reward_params": reward_params, "enable_rand_init": False}

    if motion_clips_path is not None:
        reward_params["motion_clips_file_path"] = motion_clips_path  # add reward path to reward params
    if urdf_path is not None:
        env_params["urdf_path"] = urdf_path

    if params['pretrained_file'] is not None:
        pretrained_clip_file = params['pretrained_file']
        pretrained_clip_file = os.path.join("data", "deepmimic", "pretrained", pretrained_clip_file)
        reward_params["pretrained_clips_file_path"] = pretrained_clip_file
        
    # =============
    # create a simple environment for evaluation
    # =============

    # eval_env = gym.make(env_id, **env_kwargs)

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv
    )
    
    # =============
    # start playing
    # =============
    episodes = 10000
    frame_rate = 60
    
    for ep in range(episodes):
        done1 = False
        # done2 = False
        # t1 = eval_env.envs[0].phase*eval_env.envs[0].dataset.duration
        # t2 = eval_env.envs[1].phase*eval_env.envs[1].dataset.duration
        phase = 0
        while phase <= 2:
            phase += (1/frame_rate)/eval_env.envs[0].motion.duration
            if phase <= 2: 
                eval_env.envs[0].reset(phase = phase)
            else:
                action = eval_env.envs[0].joint_angle_default
                eval_env.envs[0]._sim.step(action)

            eval_env.envs[0].render("human")
            # t1 += 1.0/(frame_rate)
            # t2 += 1.0/frame_rate
            # time.sleep(0.01)
    eval_env.close()


