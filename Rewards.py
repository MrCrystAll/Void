import json
import os
from typing import Tuple

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward


class TraceableSB3CombinedLogReward(SB3CombinedLogReward):
    def __init__(self, reward_functions: Tuple[RewardFunction, ...], reward_weights: Tuple[float, ...], reward_update):
        super().__init__(reward_functions, reward_weights)

        self.reward_update = reward_update

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.returns += [a * float(b) for a, b in zip(rewards, self.reward_weights)]  # store the rewards
        if self.reward_update:
            self.reward_update(player, self.returns)

        return float(np.dot(self.reward_weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        # Add the rewards to the cumulative totals with numpy broadcasting
        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]
        if self.reward_update:
            self.reward_update(player, self.returns)

        # Obtain the lock
        while True:
            try:
                open(self.lockfile, 'x')
                break
            except FileExistsError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f'Error obtaining lock in SB3CombinedLogReward.get_final_reward:\n{e}')

        # Write the rewards to file and reset
        with open(self.file_location, 'a') as f:
            f.write('\n' + json.dumps(self.returns.tolist()))

        # reset the episode totals
        self.returns = np.zeros(len(self.reward_functions))

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

        return float(np.dot(self.reward_weights, rewards))