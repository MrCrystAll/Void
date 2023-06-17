import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_RADIUS
from rlgym_sim.utils.gamestates import GameState, PlayerData


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