import argparse
import json
import logging
import os

from tqdm import tqdm

from pylocogym.envs.video_recoder import VecVideoRecorder


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from pylocogym.algorithms import CustomPPO

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        params = json.load(f)

    export_params = params["export"]
    model_file = export_params["model_file"]
    env_id = params["env_id"]
    hyp_params = params["train_hyp_params"]
    env_params = params['environment_params']
    reward_params = params['reward_params']
    venv_file = "model_data/walk_task/vecnormalize_34000000_steps.pkl"

    max_episode_steps = hyp_params.get("max_episode_steps", 5000)
    seed = hyp_params.get("seed", 313)
    env_kwargs = {"max_episode_steps": max_episode_steps, 
                  "env_params": env_params, 
                  "reward_params": reward_params,
                  "enable_rand_init": False}

    motion_clip_file = params['motion_file']
    motion_clip_file = os.path.join("data", "deepmimic", "motions", motion_clip_file)
    reward_params["motion_clips_file_path"] = motion_clip_file

    eval_env = make_vec_env(
        env_id,
        n_envs=hyp_params["num_envs"],
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )
    if hyp_params.get("normalize_observation") or hyp_params.get("normalize_reward"):
        eval_env = VecNormalize.load(venv_file,eval_env)
    eval_env = VecVideoRecorder(
        eval_env,
        topk=export_params["topk"],
    )
    

    model = CustomPPO.load(model_file, eval_env)

    obs = eval_env.reset()
    for ep in tqdm(range(export_params["max_steps"])):
        action, _ = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        # eval_env.render()
    input_fps = env_params["control_rate"]

    eval_env.close()
    export_dir = export_params["out_dir"]
    fps = export_params["fps"]
    eval_env.save_videos(export_dir, export_params.get("input_fps", input_fps), fps)
