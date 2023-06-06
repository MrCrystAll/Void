import os
import warnings

from redis import Redis
from rlgym_sim.envs import Match
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker

from StateSetters import ProbabilisticStateSetter
from config import Configuration, version_dict

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

    def create_match(version):

        if version not in version_dict:
            warnings.warn(f"Version '{version}' is not in the config dictionary, "
                          f"possible keys are {list(version_dict.keys())}, switching version to default")
            version = "default"

        match_config: Configuration = version_dict[version]

        return Match(
            team_size=match_config.team_size,
            reward_function=SB3CombinedLogReward(
                reward_functions=match_config.rewards[0],
                reward_weights=match_config.rewards[1]
            ),
            spawn_opponents=match_config.spawn_opponents,
            terminal_conditions=match_config.terminal_conditions,
            obs_builder=match_config.obs_builder,
            state_setter=ProbabilisticStateSetter(
                verbose=1,
                states=match_config.state_setter[0],
                probs=match_config.state_setter[1]
            ),
            action_parser=match_config.action_parser,  # Discrete > Continuous don't @ me
        )

    match = create_match("aerial")

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
