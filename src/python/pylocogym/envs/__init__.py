from gym.envs.registration import register
from .VanillaEnv import VanillaEnv
from .ResidualEnv import ResidualEnv

register(id="PylocoVanilla-v0", entry_point=VanillaEnv)
register(id="PylocoResidual-v0", entry_point=ResidualEnv)