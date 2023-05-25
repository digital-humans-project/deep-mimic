import os.path
import sys
import json
import matplotlib
matplotlib.use("Agg")

from pylocogym.cmake_variables import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


if __name__ == "__main__":

    motion_clip_file = "humanoid3d_jump.txt"
    config = "bob_env.json"
    log_path = PYLOCO_LOG_PATH
    data_path = PYLOCO_DATA_PATH
    model_file = "model_data/jump/model_24000000_steps.zip"

    # config file
    if config is None:
        sys.exit('Config name needs to be specified for training: --config <config file name>')
    else:
        config_path = os.path.join(data_path, 'conf', config)
        print('- config file path = {}'.format(config_path))

    with open(config_path, 'r') as f:
        params = json.load(f)

    env_id = params['env_id']
    env_params = params['environment_params']
    hyp_params = params['train_hyp_params']
    reward_params = params['reward_params']

    max_episode_steps = hyp_params.get('max_episode_steps', 5000)
    seed = hyp_params.get("seed", 313)
    env_kwargs = {"max_episode_steps": max_episode_steps, 
                  "env_params": env_params, 
                  "reward_params": reward_params,
                  "enable_rand_init": False}
    
    if motion_clip_file is not None:
        motion_clip_file = os.path.join("data", "deepmimic", "motions", motion_clip_file)
        reward_params["motion_clips_file_path"] = motion_clip_file

    # =============
    # create a simple environment for testing model
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
    # Load pre-trained model
    # =============
    model = PPO.load(model_file, eval_env)

    # =============
    # start playing
    # =============
    episodes = 10000
    frame_rate = 60
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            eval_env.envs[0].render("human")
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.envs[0].step(action.squeeze())
            print("now phase", obs[-1])
    eval_env.close()