import random

import numpy as np
from rlgym_sim.envs import Match as SimMatch
from rlgym.envs import Match as Match


class DynamicGMMatchSim(SimMatch):
    def __init__(self, reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                 spawn_opponents, gm_weights):
        super().__init__(reward_function, terminal_conditions, obs_builder, action_parser, state_setter, team_size,
                         spawn_opponents)

        assert team_size == len(
            gm_weights), f"There is a maximum of {team_size} players in a team, but there are {len(gm_weights)} game modes"
        self.gm_weights = gm_weights

    def pre_reset(self):
        team_size = random.choices(range(1, self.team_size + 1), weights=self.gm_weights, k=1)[0]
        self.agents = team_size * 2 if self.spawn_opponents else team_size

    def get_reset_state(self) -> list:
        new_state = self._state_setter.build_wrapper(self.agents // 2 if self.spawn_opponents else self.agents, self.spawn_opponents)
        self._state_setter.reset(new_state)
        print(f"Resetting with {self.agents} agents")

        return new_state.format_state()


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
