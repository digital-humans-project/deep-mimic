from gym.envs.registration import register
from .VanillaEnv import VanillaEnv
from .ResidualEnv_v1 import ResidualEnv_v1

register(id="PylocoVanilla-v0", entry_point=VanillaEnv)
register(id="ResidualEnv-v1", entry_point=ResidualEnv_v1)
