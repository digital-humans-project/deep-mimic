import pathlib
import datetime
import time
from stable_baselines3.common import utils
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


def manage_save_path(log_dir, name):
    date = datetime.date.today().strftime("%Y-%m-%d-")
    name = date + name
    latest_run_id = utils.get_latest_run_id(log_dir, name)
    save_path = utils.os.path.join(log_dir, f"{name}_{latest_run_id + 1}")
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    return save_path


def play(params, log_path, dir_name, reward_path=None):
    """Render environment using given action"""

    save_path = manage_save_path(log_path, dir_name)

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
    # create a single environment for evaluation
    # =============

    # Documentation: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    # Useful to have the documentation open for the subsequent calls of the "eval_env" as well.
    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )

    # =============
    # start playing
    # =============
    episodes = 100
    action_shape = eval_env.action_space.shape[0] 
    for ep in range(episodes):
        # "eval_env.reset()"
        # Reset all the environments and return an array of observations, or a tuple of observation arrays.
        # If step_async is still doing work, that work will be cancelled and step_wait() should not be called until step_async() is invoked again.
        # Return type : Union[ndarray, Dict[str, ndarray], Tuple[ndarray, ...]]
        # Returns: observation

        # ? (Konstantinos)
        # Resetting the environment does not reset the computed policy as well, right?
        # Therefore, the "initial condition" of the system (e.g. Bob standing upright) is the same, but the policy has evolved.
        eval_env.reset()
        done = False
        # action = eval_env.action_space.sample()*0.5  # 0.5 to avoid big angle change
        action = np.zeros(action_shape) # zero point visualization
        action[21] = -0.8
        print(action)
        while not done:
            eval_env.render("human")
            obs, reward, done, info = eval_env.step([action])
        time.sleep(0.1)
            
    eval_env.close()
