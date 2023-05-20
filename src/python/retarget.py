import pathlib
import datetime
import time
import gym
from pylocogym.data.deep_mimic_bob_adapter import BobMotionBobAdapter, BobMotionDataFieldNames
from pylocogym.data.deep_mimic_motion import DeepMimicMotion
from pylocogym.data.lerp_dataset import LerpMotionDataset
from pylocogym.data.loop_dataset import LoopKeyframeMotionDataset
from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from scipy.spatial.transform import Rotation as R


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
    # create a simple environment for evaluation
    # =============

    # eval_env = gym.make(env_id, **env_kwargs)

    eval_env = make_vec_env(
        env_id,
        n_envs=2,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv
    )
    
    # =============
    # start playing
    # =============
    episodes = 100
    frame_rate = 30

    for ep in range(episodes):
        eval_env.reset()
        done1 = False
        done2 = False
        t1 = eval_env.envs[0].phase*eval_env.envs[0].dataset.duration
        t2 = eval_env.envs[1].phase*eval_env.envs[1].dataset.duration
        while not done1 and not done2:
            eval_env.envs[0].render("human")
            action = eval_env.envs[0].lerp.eval(t1).q[6:]
            obs, reward, done1, info = eval_env.envs[0].step(action)
            print("now time, now phase", obs[-2],obs[-1])

            action2 = eval_env.envs[1].lerp.eval(t2).q[6:]
            _,_,done2,_ = eval_env.envs[1].step(action2)

            t1 += 1.0/frame_rate
            t2 += 1.0/frame_rate
            # time.sleep(0.1)
    eval_env.close()
