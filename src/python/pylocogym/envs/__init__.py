from gym.envs.registration import register
from .VanillaEnv import VanillaEnv
from .TaskEnv import TaskEnv
from .MultiClipEnv import MultiClipEnv

register(id="PylocoVanilla-v0", entry_point=VanillaEnv)
register(id="PylocoVanillaTask-v0", entry_point=TaskEnv)
register(id="PylocoMultiClip-v0", entry_point=MultiClipEnv)
