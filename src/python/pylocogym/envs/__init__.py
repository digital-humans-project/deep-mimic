from gym.envs.registration import register
from .VanillaEnv import VanillaEnv
from .TaskEnv import TaskEnv
from .ResidualEnv_v0 import ResidualEnv_v0
from .ResidualEnv_v1 import ResidualEnv_v1
from .MultiClipEnv import MultiClipEnv

register(id="PylocoVanilla-v0", entry_point=VanillaEnv)
register(id="PylocoVanillaTask-v0", entry_point=TaskEnv)
register(id="ResidualEnv-v0", entry_point=ResidualEnv_v0)
register(id="ResidualEnv-v1", entry_point=ResidualEnv_v1)
register(id="PylocoMultiClip-v0", entry_point=MultiClipEnv)
