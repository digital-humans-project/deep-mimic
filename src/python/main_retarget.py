import os.path
import sys
import json
import matplotlib
matplotlib.use("Agg")

from src.python import retarget
from pylocogym.cmake_variables import *


if __name__ == "__main__":

    #motion_clip_file = "humanoid3d_jog.txt"
    #motion_clip_file = "jogwalk_multiclip.txt"
    motion_clip_file = "amass/humanoid3d_walk_to_ready_mirror.txt"
    #motion_clip_file = "sie/sie_humanoid3d_punch_forward02.txt"
    config = "bob_env_walk.json"
    # urdf_file = "data/robots/deep-mimic/humanoid.urdf"

    # log path
    log_path = PYLOCO_LOG_PATH
    data_path = PYLOCO_DATA_PATH

    if motion_clip_file is not None:
        motion_clip_file = os.path.join("data", "deepmimic", "motions", motion_clip_file)

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
    dir_name = "{id}-{rew}-{steps:.1f}M".format(id=params['env_id'], rew=motion_clip_file, steps=float(steps / 1e6))

    # Play environment
    retarget.play_motion(
        params=params,
        motion_clips_path=motion_clip_file,
    )