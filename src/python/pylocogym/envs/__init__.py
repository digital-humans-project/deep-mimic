from gym.envs.registration import register
from .VanillaEnv import VanillaEnv
from .TaskEnv import TaskEnv

register(id="PylocoVanilla-v0", entry_point=VanillaEnv)
register(id="PylocoVanillaTask-v0", entry_point=TaskEnv)