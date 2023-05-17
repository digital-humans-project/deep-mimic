import os.path
import sys
import json
import matplotlib
matplotlib.use("Agg")

import visualizer
from pylocogym.cmake_variables import *


if __name__ == "__main__":

    rewardFile = "./bob/humanoid_reward.py"
    config = "bob_env.json"

    # log path
    log_path = PYLOCO_LOG_PATH
    data_path = PYLOCO_DATA_PATH

    rewardFile_formatted = rewardFile.replace(".py", "").replace("./","").replace("/","_")
    if rewardFile is not None:
        # Heavily assumes the directory structure of the project. Be careful to include the reward file in the predertermined path below.
        rewardFile = os.path.join("src", "python", "pylocogym", "envs", "rewards", rewardFile)

    # config file
    if config is None:
        sys.exit('Config name needs to be specified for training: --config <config file name>')
    else:
        config_path = os.path.join(data_path, 'conf', config)
        print(f'- config file path = {config_path}')

    with open(config_path, 'r') as f:
        params = json.load(f)

    # key parameters
    hyp_params = params['train_hyp_params']
    steps = hyp_params['time_steps']
    dir_name = "{id}-{rew}-{steps:.1f}M".format(id=params['env_id'], rew=rewardFile_formatted, steps=float(steps / 1e6))

    # Play environment
    visualizer.play(
        params=params,
        log_path=log_path,
        dir_name=dir_name,
        reward_path=rewardFile
    )