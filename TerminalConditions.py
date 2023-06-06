import warnings
from typing import Iterable

import rlgym_sim.utils.common_values
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    BallTouchedCondition

class BallGroundCondition(TerminalCondition):

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.ball.position.item(2) < rlgym_sim.utils.common_values.BALL_RADIUS + 100

class BallTouchedAfterSteps(TerminalCondition):

    def __init__(self, steps_before_reset: int):
        super().__init__()
        self.steps_before_reset = steps_before_reset
        self.current_steps = 0
        self.ball_touched = False

    def reset(self, initial_state: GameState):
        self.current_steps = 0
        self.ball_touched = False

    def is_terminal(self, current_state: GameState) -> bool:

        if self.ball_touched:
            self.current_steps += 1

        else:
            for player in current_state.players:
                if player.ball_touched:
                    self.ball_touched = True
                    break

        return self.ball_touched and self.current_steps >= self.steps_before_reset