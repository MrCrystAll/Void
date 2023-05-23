import os
from typing import Any

import numpy
from redis import Redis
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from rlgym.utils.gamestates import PlayerData
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, TouchBallReward, \
    FaceBallReward
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_sim.envs import Match
from rlgym_sim.utils.gamestates import GameState
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker

from AstraObs import AstraObs
from StateSetters import ProbabilisticStateSetter


class ExpandAdvancedObs(AstraObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the 
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch

    torch.set_num_threads(1)

    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    match = Match(
        spawn_opponents=True,
        team_size=3,
        state_setter=ProbabilisticStateSetter(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DiscreteAction(),
        terminal_conditions=[TimeoutCondition(round(2000)),
                             GoalScoredCondition()],
        reward_function=SB3CombinedLogReward(
            reward_functions=(
                VelocityBallToGoalReward(),
                EventReward(
                    goal=100,
                    concede=-100,
                    save=50,
                ),
                FaceBallReward(),
                TouchBallReward(),
            ),
            reward_weights=(1.0, 2.0, 1.0, 1.0)),
    )

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", username="test-bot", password=os.environ["REDIS_PASSWORD"], port=6379, db=3)

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(r, "astra_512_neurons_3_hidden_tuned_state_setter", match,
                       past_version_prob=.2,
                       evaluation_prob=0.01,
                       sigma_target=2,
                       dynamic_gm=True,
                       send_obs=True,
                       auto_minimize=False,
                       streamer_mode=False,
                       send_gamestates=False,
                       force_paging=False,
                       local_cache_name="astra_512_neurons_3_hidden_tuned_state_setter_model_db").run()
