import os
import time

import rlgym_sim
from rlgym.gamelaunch import LaunchPreference
from rlgym_sim.gym import Gym as SimGym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward

from rlgym.gym import Gym

from MyPPO import MyPPO
from StateSetters import ProbabilisticStateSetter
from config import version_dict, Configuration
from match import DynamicGMMatchSim, DynamicGMMatchGym

version = "recovery"

env_config: Configuration = version_dict[version]

match = DynamicGMMatchGym(
    state_setter=ProbabilisticStateSetter(
        states=env_config.state_setter[0],
        probs=env_config.state_setter[1]
    ),
    action_parser=env_config.action_parser,
    obs_builder=env_config.obs_builder,
    team_size=env_config.team_size,
    spawn_opponents=env_config.spawn_opponents,
    terminal_conditions=env_config.terminal_conditions,
    reward_function=ConstantReward(),
    gm_weights=[0.33]
)

env = Gym(match, os.getpid(), launch_preference=LaunchPreference.STEAM)


def LoadModel():
    return MyPPO.load("models/exit_save.zip", env=env)


model_time = 320
obs = env.reset()

try:
    model = LoadModel()
    obs = env.reset()
    start_time = time.perf_counter()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, terminated, info = env.step(action)

        if terminated:
            # model = LoadModel()
            start_time = time.perf_counter()
            obs = env.reset()

        if start_time + model_time < time.perf_counter():
            model = LoadModel()
            obs = env.reset()
            start_time = time.perf_counter()

except KeyboardInterrupt:
    print("Exiting")


