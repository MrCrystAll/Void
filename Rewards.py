import json
import os
from json import JSONEncoder
from typing import Tuple, Optional

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_RADIUS
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import CombinedReward


class KickOffReward:
    pass


class DistancePlayerToBall(RewardFunction):
    def __init__(self):
        self.data = []

    def reset(self, initial_state: GameState):
        if len(self.data) != 0:
            self.data.clear()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        self.data.append(dist)

        return -(dist ** 2) / 6_000_000 + 0.5


class EpisodeLengthReward(RewardFunction):
    def __init__(self):
        self.nb_steps_since_reset = 0

    def reset(self, initial_state):
        self.nb_steps_since_reset = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.nb_steps_since_reset += 1

        return - self.nb_steps_since_reset ** 2 / 25_000 + 1

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


class ObservableSB3CombinedLogReward(CombinedReward):

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None,
            file_location: str = 'combinedlogfiles'
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward. Will also log the weighted rewards to
        the model's logger if a SB3CombinedLogRewardCallback is provided to the
        learner.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        :param file_location: The path to the directory that will be used to
        transfer reward info
        """
        super().__init__(reward_functions, reward_weights)

        # Make sure there is a folder to dump to
        os.makedirs(file_location, exist_ok=True)
        self.file_location = f'{file_location}/rewards.txt'
        self.lockfile = f'{file_location}/reward_lock'

        # Initiates the array that will store the episode totals
        self.returns = np.zeros(len(self.reward_functions))
        self.all_ep_rewards = {}

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
                print(f'Error obtaining lock in SB3CombinedLogReward.__init__:\n{e}')

        # Empty the file by opening in w mode
        with open(self.file_location, 'w') as f:
            pass

        # Release the lock
        try:
            os.remove(self.lockfile)
        except FileNotFoundError:
            print('No lock to release! ')

    def reset(self, initial_state: GameState):

        if self.all_ep_rewards != {}:
            rewards_dict = {}
            rewards_list = []
            final_rewards = {}

            with open("ui/rewards.json", "w") as f:
                for key in self.all_ep_rewards.keys():
                    rewards_dict.setdefault(key, np.sum(self.all_ep_rewards[key], axis=0).tolist())
                    rewards_list.append(rewards_dict[key])

                rewards_list = np.mean(rewards_list, axis=0).tolist()

                for index, reward in enumerate(self.reward_functions):
                    final_rewards.setdefault(reward.__class__.__name__, rewards_list[index])

                f.write(JSONEncoder().encode(final_rewards))

        self.returns = np.zeros(len(self.reward_functions))
        self.all_ep_rewards.clear()
        super().reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player in state.players:
            rewards = [
                func.get_reward(player, state, previous_action)
                for func in self.reward_functions
            ]

            if player.car_id not in self.all_ep_rewards:
                self.all_ep_rewards.setdefault(player.car_id, [])

            self.all_ep_rewards[player.car_id].append([a * float(b) for a, b in zip(rewards, self.reward_weights)])

            self.returns += [a * float(b) for a, b in zip(rewards, self.reward_weights)]  # store the rewards

            return float(np.dot(self.reward_weights, rewards))
        else:
            self.returns += [0 * len(self.reward_functions)]
            if player.car_id not in self.all_ep_rewards:
                self.all_ep_rewards.setdefault(player.car_id, [])

            self.all_ep_rewards[player.car_id].append([0 * len(self.reward_functions)])

            return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        if player.car_id not in self.all_ep_rewards:
            self.all_ep_rewards.setdefault(player.car_id, [])

        self.all_ep_rewards[player.car_id].append([a * b for a, b in zip(rewards, self.reward_weights)])

        # Add the rewards to the cumulative totals with numpy broadcasting
        self.returns += [a * b for a, b in zip(rewards, self.reward_weights)]



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
