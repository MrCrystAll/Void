import random
from typing import List, Union

import numpy as np
from rlgym_sim.envs import Match as SimMatch
from rlgym.envs import Match as Match

from Rewards import ObservableSB3CombinedLogReward
from ui.UIData import UIData


class DynamicGMMatchSim(SimMatch):
    def __init__(self, reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                 spawn_opponents, gm_weights):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         spawn_opponents)

        assert team_size == len(
            gm_weights), f"There is a maximum of {team_size} players in a team, but there are {len(gm_weights)} game modes"
        self.gm_weights = gm_weights

    def get_reset_state(self) -> list:
        team_size = random.choices(range(1, self.team_size + 1), weights=self.gm_weights, k=1)[0]
        self.agents = team_size * 2 if self.spawn_opponents else team_size
        new_state = self._state_setter.build_wrapper(team_size, self.spawn_opponents)
        self._state_setter.reset(new_state)
        print(f"Resetting with {self.agents} agents")

        return new_state.format_state()

    def format_actions(self, actions: np.ndarray):
        self._prev_actions[:len(self._spectator_ids)] = actions[:len(self._spectator_ids)]
        acts = []
        for i in range(len(self._spectator_ids)):
            acts.append(float(self._spectator_ids[i]))
            for act in actions[i]:
                acts.append(float(act))

        return acts


class ObservableMatch(SimMatch):
    def __init__(self, reward_function: ObservableSB3CombinedLogReward, terminal_conditions, obs_builder, action_parser,
                 state_setter, team_size,
                 spawn_opponents):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         spawn_opponents)
        self.ui_data = UIData()
        self.monitored = False

    def get_reset_state(self) -> list:

        if self.ui_data.state_name and self.ui_data.rewards:
            print("Got everything, writing in data.json")
            with open("ui/data.json", "w") as f:
                f.write(self.ui_data.jsonify())

            print("Wrote in ui/data.json")
            self.ui_data = UIData()

        new_state = self._state_setter.build_wrapper(self.team_size, self.spawn_opponents)
        name = self._state_setter.reset(new_state, True)
        self.ui_data.state_name = name
        print(f"Resetting with {self.agents}")

        return new_state.format_state()

    def get_rewards(self, state, done) -> Union[float, List]:

        rewards = super().get_rewards(state, done)

        if done:
            self.ui_data.rewards = self._reward_fn.calculate_rew_for_players()
        return rewards


class ObservableDynamicGMMatchSim(DynamicGMMatchSim):

    def __init__(self, reward_function: ObservableSB3CombinedLogReward, terminal_conditions, obs_builder, action_parser,
                 state_setter, team_size,
                 spawn_opponents, gm_weights):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         spawn_opponents, gm_weights)
        self.ui_data = UIData()
        self.monitored = False

    def get_reset_state(self) -> list:

        if self.ui_data.state_name and self.ui_data.rewards:
            with open("ui/data.json", "w") as f:
                f.write(self.ui_data.jsonify())
            self.ui_data = UIData()

        team_size = random.choices(range(1, self.team_size + 1), weights=self.gm_weights, k=1)[0]
        self.agents = team_size * 2 if self.spawn_opponents else team_size
        new_state = self._state_setter.build_wrapper(team_size, self.spawn_opponents)
        name = self._state_setter.reset(new_state, True)
        self.ui_data.state_name = name
        print(f"Resetting with {self.agents}")

        return new_state.format_state()

    def get_rewards(self, state, done) -> Union[float, List]:
        if done:
            self.ui_data.rewards = self._reward_fn.calculate_rew_for_players()

        return super().get_rewards(state, done)


class DynamicGMMatchGym(Match):
    def __init__(self, reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                 spawn_opponents, gm_weights):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         spawn_opponents=spawn_opponents)
        assert team_size == len(
            gm_weights), f"There is a maximum of {team_size} players in a team, but there are {len(gm_weights)} game modes"
        self.gm_weights = gm_weights

    def get_reset_state(self) -> list:
        team_size = random.choices(range(1, self._team_size + 1), weights=self.gm_weights, k=1)[0]
        self.agents = team_size * 2 if self._spawn_opponents else team_size
        new_state = self._state_setter.build_wrapper(team_size, self._spawn_opponents)

        self._state_setter.reset(new_state)

        return new_state.format_state()

    def format_actions(self, actions: np.ndarray):
        self._prev_actions[:len(self._spectator_ids)] = actions[:len(self._spectator_ids)]
        acts = []
        for i in range(len(self._spectator_ids)):
            acts.append(float(self._spectator_ids[i]))
            for act in actions[i]:
                acts.append(float(act))

        return acts
