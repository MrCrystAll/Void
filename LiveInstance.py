import time

import rlgym_sim
from rlgym_sim.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward

from MyPPO import MyPPO
from StateSetters import ProbabilisticStateSetter
from config import version_dict, Configuration
from match import DynamicGMMatch

version = "default"

env_config: Configuration = version_dict[version]

match = DynamicGMMatch(
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
    gm_weights=[0.1, 0.8, 0.1]
)

env = Gym(match, tick_skip=8, gravity=1, boost_consumption=1, dodge_deadzone=0.8, copy_gamestate_every_step=False)


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
            env.reset()

        if start_time + model_time < time.perf_counter():
            model = LoadModel()
            obs = env.reset()
            start_time = time.perf_counter()

except KeyboardInterrupt:
    print("Exiting")


