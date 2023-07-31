from typing import Dict, List, Tuple, Any, Union

import numpy as np
from rlgym_sim.gym import Gym


class ObservableGym(Gym):

    def render(self, mode="human"):
        pass

    def __init__(self, match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption):
        super().__init__(match, copy_gamestate_every_step, dodge_deadzone, tick_skip, gravity, boost_consumption)

    def pre_reset(self):
        self._match.pre_reset()

    def reset(self, return_info=False) -> Union[List, Tuple]:
        """
        The environment reset function. When called, this will reset the state of the environment and objects in the game.
        This should be called once when the environment is initialized, then every time the `done` flag from the `step()`
        function is `True`.
        """

        state_str = self._match.get_reset_state()
        state = self._game.reset(state_str)

        self._match.episode_reset(state)
        self._prev_state = state

        obs = self._match.build_observations(state)
        if return_info:
            info = {
                'state': state,
                'result': self._match.get_result(state),
                'team_size': self._match.team_size,
                'so': self._match.spawn_opponents
            }
            return obs, info
        return obs

    def step(self, actions: Any) -> Tuple[List, List, bool, Dict]:
        """
        The step function will send the list of provided actions to the game, then advance the game forward by `tick_skip`
        physics ticks using that action. The game is then paused, and the current state is sent back to rlgym_sim This is
        decoded into a `GameState` object, which gets passed to the configuration objects to determine the rewards,
        next observation, and done signal.

        :param actions: An object containing actions, in the format specified by the `ActionParser`.
        :return: A tuple containing (obs, rewards, done, info)
        """

        actions = self._match.format_actions(self._match.parse_actions(actions, self._prev_state))

        state = self._game.step(actions)

        obs = self._match.build_observations(state)
        done = self._match.is_done(state)
        reward = self._match.get_rewards(state, done)
        self._prev_state = state

        info = {
            'prev_state': self._prev_state,
            'state': state,
            'result': self._match.get_result(state),
            'team_size': self._match.team_size,
            'so': self._match.spawn_opponents
        }

        return obs, reward, done, info