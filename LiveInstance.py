import os
import pickle

import rlgym
import torch
from redis.client import Redis
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward

from StateSetters import ProbabilisticStateSetter
from config import version_dict, Configuration

version = "recovery"

env_config: Configuration = version_dict[version]

r = Redis(host="127.0.0.1", username="test-bot", password=os.environ["REDIS_PASSWORD"], port=6379, db=0)

model = r.get("model-latest")
model = pickle.loads(model)

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
obs = torch.Tensor(obs)

while True:
    try:
        action_space = model.get_action_distribution(obs)
        action = model.sample_action(action_space, deterministic=True)
        action = action.numpy()

        obs, _, terminal, _ = env.step(action)
        obs = torch.Tensor(obs)

        if terminal:
            obs = env.reset()
            obs = torch.Tensor(obs)

    except KeyboardInterrupt:
        break


