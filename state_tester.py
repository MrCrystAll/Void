import numpy as np
import rlviser_py
from rlgym_sim.envs import Match
from rlgym_sim.gym import Gym
from rlgym_sim.utils import StateSetter
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from stable_baselines3 import PPO

from Rewards import RewardTracker
from StateSetters import ProbabilisticStateSetter, DefaultState
from config import Configuration, version_dict


class StateTester:
    def __init__(
            self,
            env: Gym,
            state: StateSetter,
            learner: PPO,
            reward: RewardTracker,
            render: bool = True,
            deterministic: bool = True
            ):
        self.env = env
        self.state = state
        self.learner = learner
        self.reward_tracker = reward
        self.render = render
        self.deterministic = deterministic

        self._last_obs = None
        self.state = None
        self._init_env()

    def _init_env(self):
        self._last_obs, info = self._reset()
        self.state = info["state"]

    def launch_sample(self, n_episodes: int = 1):
        current_ep = 0
        while current_ep < n_episodes:
            self._render() if self.render else None

            actions = self.learner.policy.predict(np.array(self._last_obs, dtype=object), deterministic=self.deterministic)
            new_obs, rewards, terminal, info = self._step(actions)

            # Do something with the rewards
            self._register_reward(rewards)

            if terminal:
                current_ep += 1
                new_obs, info = self._reset()

            self.state = info["state"]
            self._last_obs = new_obs


    def _step(self, actions):
        return self.env.step(actions)

    def _reset(self):
        return self.env.reset(return_info=True)

    def _render(self):
        rlviser_py.render_rlgym(self.state)

    def _register_reward(self, rewards):
        print(rewards)
        print(self.reward_tracker.calculate_rew_for_players())


if __name__ == "__main__":
    match_config: Configuration = version_dict["default"]

    env = Match(
            team_size=match_config.team_size,
            reward_function=SB3CombinedLogReward(
                reward_functions=match_config.rewards[0],
                reward_weights=match_config.rewards[1]
            ),
            spawn_opponents=match_config.spawn_opponents,
            terminal_conditions=match_config.terminal_conditions,
            obs_builder=match_config.obs_builder,
            state_setter=ProbabilisticStateSetter(
                verbose=1,
                states=match_config.state_setter[0],
                probs=match_config.state_setter[1]
            ),
            action_parser=match_config.action_parser
        )

    env = Gym(match=env, tick_skip=8, dodge_deadzone=0.8, gravity=1, copy_gamestate_every_step=True, boost_consumption=1)

    StateTester(
        env=env,
        reward=RewardTracker(SB3CombinedLogReward(
                reward_functions=match_config.rewards[0],
                reward_weights=match_config.rewards[1]
            )),
        learner=PPO.load("models/exit_save", env=env, custom_objects={
            "num_envs": 1
        }),
        state=DefaultState()
    ).launch_sample(2)