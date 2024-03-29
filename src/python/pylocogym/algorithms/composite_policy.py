from typing import List

import torch
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from torch import Tensor, nn


class ComposeActorCriticPolicy(BasePolicy):
    def __init__(self, policies: List[ActorCriticPolicy], tempreture=0.3):
        super().__init__(observation_space=policies[0].observation_space, action_space=policies[0].action_space)
        self.policies = nn.ModuleList(policies)
        self.tempreture = tempreture

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _predict(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        values = [p.predict_values(observation) for p in self.policies]
        actions = [p.get_distribution(observation).get_actions(deterministic=deterministic) for p in self.policies]
        values = torch.stack(values)
        actions = torch.stack(actions)
        weights = (values / self.tempreture).softmax(dim=0)
        action = (actions * weights).sum(dim=0)
        return action
