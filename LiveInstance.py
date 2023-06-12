import os
import pickle

import numpy as np
import rlgym
import rlgym_sim
import torch
from redis.client import Redis
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from stable_baselines3.common.utils import obs_as_tensor

from MyPPO import MyPPO
from StateSetters import ProbabilisticStateSetter
from config import version_dict, Configuration

version = "recovery"

env_config: Configuration = version_dict[version]

# If it can't load, let it crash, don't create a new model
model = MyPPO.load("models/exit_save.zip")

env = rlgym.make(
    game_speed=1,
    state_setter=ProbabilisticStateSetter(
        states=env_config.state_setter[0],
        probs=env_config.state_setter[1]
    ),
    action_parser=env_config.action_parser,
    obs_builder=env_config.obs_builder,
    team_size=env_config.team_size,
    spawn_opponents=env_config.spawn_opponents,
    terminal_conditions=env_config.terminal_conditions,
    tick_skip=8,
    reward_fn=ConstantReward(),
)

obs = env.reset()

while True:
    try:
        actions, _, _ = model.policy(obs_as_tensor(np.array(obs), torch.device("cuda")))
        actions = actions.cpu().numpy()

        obs, _, terminal, _ = env.step(actions)
        obs = torch.Tensor(obs)

        if terminal:
            obs = env.reset()
            obs = torch.Tensor(obs)

    except KeyboardInterrupt:
        break


