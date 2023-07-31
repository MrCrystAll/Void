from typing import Any

import numpy as np
import rlgym.utils.common_values as constants
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import AdvancedObs


class AstraObs(AdvancedObs):
    BOOST_STD = 10

    def __init__(self, team_size=3, tick_skip: int = 8):
        super().__init__()
        self.nb_players = team_size * 2  # 2 teams
        self.tick_skip = tick_skip
        self.time_interval = self.tick_skip / 120  # Running at 120fps

        # Timers start at 10 (refresh time) and end up at 0
        self.boosts_timers = np.zeros(len(constants.BOOST_LOCATIONS))
        self.boosts_state = np.zeros(len(constants.BOOST_LOCATIONS))
        self.boosts_location = np.array(constants.BOOST_LOCATIONS)
        self.inverted_boosts_state = self.boosts_state[::-1]
        self.inverted_boosts_timers = self.boosts_timers[::-1]
        self.last_boost_amount = {}
        self.ball_prediction = None
        self.lookahead_steps = 5

    def pre_step(self, state: GameState):
        # Boost timer
        self._update_timers(state)
        self.boosts_timers /= self.BOOST_STD
        self.inverted_boosts_timers /= self.BOOST_STD

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == constants.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        allies = []
        enemies = []

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        team_obs = []
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies

            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        base_obs = np.concatenate(obs)

        # Populate with players
        missed_players = self.nb_players - len(state.players)

        for i in range(missed_players):
            base_obs = self.create_dummy_players(base_obs)

        if player.team_num == constants.ORANGE_TEAM:
            timers = self.inverted_boosts_timers
        else:
            timers = self.boosts_timers

        base_obs = np.concatenate((base_obs, timers))

        base_obs = np.nan_to_num(base_obs, nan=0)
        self.post_step(state)

        return base_obs

    def post_step(self, state: GameState):
        for car in state.players:
            if car.car_id in self.last_boost_amount:
                self.last_boost_amount[car.car_id] = car.boost_amount
            else:
                self.last_boost_amount.setdefault(car.car_id, car.boost_amount)

    @staticmethod
    def print_obs(obs):
        print(f"Obs length : {len(obs)}")
        print(f"Ball position :         {obs[0:3] * AstraObs.POS_STD}")
        print(f"Ball linear velocity :  {obs[3:6]}")
        print(f"Ball angular velocity : {obs[6:9]}")
        print(f"Previous action :       {obs[9:17]}")
        print(f"Pads :                  {obs[17:51]}")
        print(f"The players : ")
        AstraObs.print_player_obs(obs[51:77])

        for i in range(5):
            AstraObs.print_player_obs(obs[76 + i * 32: (76 + 32) + i * 32])

        print(f"Pads timer :                        {obs[237:271]}")

    @staticmethod
    def print_player_obs(obs):
        print(f"Relative pos to ball :  {obs[0:3]}")
        print(f"Relative vel to ball :  {obs[3:6]}")
        print(f"Position :              {obs[6:9]}")
        print(f"Forward :               {obs[9:12]}")
        print(f"Up :                    {obs[12:15]}")
        print(f"Linear velocity :       {obs[15:18]}")
        print(f"Angular velocity :      {obs[18:21]}")
        print(f"Boost amount :          {obs[21]}")
        print(f"On ground :             {bool(obs[22])}")
        print(f"Has flip :              {bool(obs[23])}")
        print(f"Demo'd :                {bool(obs[24])}")

    def _update_timers(self, state: GameState):
        pads = state.boost_pads
        locs = self.boosts_location

        for i, av in enumerate(pads):
            # Unchanged state, basically just update the timer
            if av == self.boosts_state[i]:
                if av == 0:
                    self.boosts_timers[i] = max(0, self.boosts_timers[i] - self.time_interval)

            else:
                self.boosts_state[i] = int(not bool(av))
                if av == 0:
                    # Big pads
                    if locs[i][2] == 73:
                        self.boosts_timers[i] = 10
                    # Small pads
                    else:
                        self.boosts_timers[i] = 4
                else:
                    self.boosts_timers[i] = 0

        self.boosts_state = pads
        self.inverted_boosts_timers = self.boosts_timers[::-1]
        self.inverted_boosts_state = self.boosts_state[::-1]

    def create_dummy_players(self, base_obs):
        return np.concatenate((base_obs, [0] * 31))


class ExpandAdvancedObs(AstraObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)
