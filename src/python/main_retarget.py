import os.path
import sys
import json
import matplotlib
matplotlib.use("Agg")

import retarget
from pylocogym.cmake_variables import *


if __name__ == "__main__":

    rewardFile = "./bob/humanoid_reward.py"
    config = "bob_env.json"
    dataFile = "data/motions/humanoid3d_walk.txt"

    # log path
    log_path = PYLOCO_LOG_PATH
    data_path = PYLOCO_DATA_PATH

    rewardFile_formatted = rewardFile.replace(".py", "").replace("./","").replace("/","_")
    if rewardFile is not None:
        rewardFile = os.path.join("src", "python", "pylocogym", "envs", "rewards", rewardFile)

    # config file
    if config is None:
        sys.exit('Config name needs to be specified for training: --config <config file name>')
    else:
        config_path = os.path.join(data_path, 'conf', config)
        print('- config file path = {}'.format(config_path))

    with open(config_path, 'r') as f:
        params = json.load(f)

    # key parameters
    hyp_params = params['train_hyp_params']
    steps = hyp_params['time_steps']
    dir_name = "{id}-{rew}-{steps:.1f}M".format(id=params['env_id'], rew=rewardFile_formatted, steps=float(steps / 1e6))

    # Play environment
    retarget.test(
        params=params,
        reward_path=rewardFile,
        data_path=dataFile
    )