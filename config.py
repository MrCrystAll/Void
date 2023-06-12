from typing import Union

from rlgym.utils.action_parsers import ActionParser
from rlgym_sim.utils import ObsBuilder
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, FaceBallReward, LiuDistanceBallToGoalReward, \
    VelocityBallToGoalReward, TouchBallReward, VelocityPlayerToBallReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition, BallTouchedCondition

from AstraObs import ExpandAdvancedObs, AstraObs
from Rewards import KickOffReward, EpisodeLengthReward, DistancePlayerToBall
from StateSetters import AerialBallState, SaveState, ShotState, DefaultState, CustomStateSetter, StandingBallState, \
    AirDribble2Touch, HalfFlip, Curvedash, RandomEvenRecovery, Chaindash, Walldash, Wavedash, RecoverySetter
from TerminalConditions import BallGroundCondition, BallTouchedAfterSteps

fps = 120 // 8


class Configuration:
    def __init__(self,
                 state_setter,
                 terminal_conditions,
                 rewards,
                 action_parser: ActionParser = DiscreteAction(),
                 obs_builder: ObsBuilder = AstraObs(),
                 team_size: int = 3,
                 spawn_opponents: bool = True,
                 dynamic_gm: bool = True,
                 past_version_prob: float = .2,
                 evaluation_prob: float = 0.01,
                 sigma_target: Union[int, float] = 2,
                 send_obs: bool = True,
                 auto_minimize: bool = False,
                 streamer_mode: bool = False,
                 send_gamestates: bool = False,
                 force_paging: bool = False,
                 **kwargs):
        self.dynamic_gm = dynamic_gm
        self.past_version_prob = past_version_prob
        self.evaluation_prob = evaluation_prob
        self.sigma_target = sigma_target
        self.send_obs = send_obs
        self.auto_minimize = auto_minimize
        self.streamer_mode = streamer_mode
        self.send_gamestates = send_gamestates
        self.force_paging = force_paging
        self.rewards = rewards
        self.terminal_conditions = terminal_conditions
        self.state_setter = state_setter
        self.team_size = team_size
        self.spawn_opponents = spawn_opponents
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.kwargs = kwargs


version_dict = {
    "default": Configuration(
        state_setter=[[CustomStateSetter(), DefaultState(), ShotState(), SaveState(), AerialBallState()],
                      [1, 1, 49, 49, 20]],
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        rewards=[[EventReward(goal=100, concede=-100, touch=10, save=50, shot=50), FaceBallReward(),
                  LiuDistanceBallToGoalReward(), VelocityBallToGoalReward()], [1.0, 0.7, 1.4, 1]]
    ),
    "aerial": Configuration(
        state_setter=[[AerialBallState(), StandingBallState(), AirDribble2Touch()], [24, 3, 24]],
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 40), GoalScoredCondition(),
                             BallGroundCondition()],
        rewards=[[EventReward(goal=100), TouchBallReward(aerial_weight=2.0)], [1.0, 1.0]]
    ),
    "kickoff": Configuration(
        state_setter=[[DefaultState()], [1]],
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 40), BallTouchedCondition()],
        rewards=[[KickOffReward()], [1.0]],
    ),
    "recovery": Configuration(
        state_setter=[
            [HalfFlip(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             Curvedash(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             RandomEvenRecovery(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             Chaindash(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             Walldash(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             Wavedash(zero_boost_weight=0.1, zero_ball_vel_weight=0.5),
             RecoverySetter(zero_boost_weight=0.1, zero_ball_vel_weight=0.5)
             ],
            [0.1, 0.1, 0.25, 0.25, 0, 0.15, 0.15]
        ],
        terminal_conditions=[BallTouchedAfterSteps(40), TimeoutCondition(fps * 30)],
        rewards=[[TouchBallReward(), VelocityPlayerToBallReward(), VelocityBallToGoalReward(), EpisodeLengthReward(),
                  DistancePlayerToBall()], [10, 3, .2, .03, .1]],
        team_size=1,
        dynamic_gm=False,
        spawn_opponents=False
    )

}
