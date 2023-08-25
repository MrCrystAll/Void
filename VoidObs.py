from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM
from typing import Any, Dict

import numpy as np
from rlgym.rocket_league.obs_builders import DefaultObs


class VoidObs(DefaultObs):
    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            pads = state.inverted_boost_pad_timers
        else:
            pads = state.boost_pad_timers

        obs = super()._build_obs(agent, state, shared_info)
        obs = np.concatenate((obs, pads * self.PAD_TIMER_COEF == 1))
        return obs
