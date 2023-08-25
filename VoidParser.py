from rlgym.api import ActionParser, StateType, AgentID, ActionType, EngineActionType, SpaceType
from typing import Any, Dict

import gym
import numpy as np


class VoidParser(ActionParser):
    def get_action_space(self, agent: AgentID) -> SpaceType:
        return gym.spaces.Discrete(len(self._lookup_table))

    def parse_actions(
            self,
            actions: Dict[AgentID, ActionType],
            state: StateType,
            shared_info: Dict[str, Any]) -> Dict[AgentID, EngineActionType]:
        new_actions = {}
        for agent, action in actions.items():
            new_actions.setdefault(agent, self._lookup_table[action])

        return new_actions

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        # Don't really care
        pass

    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions
