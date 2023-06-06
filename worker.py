from multiprocessing import Process
import os
import warnings

from redis import Redis
from rlgym_sim.envs import Match
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker

from StateSetters import ProbabilisticStateSetter
from config import Configuration, version_dict

r = Redis(host="127.0.0.1", username="test-bot", password=os.environ["REDIS_PASSWORD"], port=6379, db=4)


def target(match, config: Configuration):
    RedisRolloutWorker(r, "artemis", match,
                       past_version_prob=config.past_version_prob,
                       evaluation_prob=config.evaluation_prob,
                       sigma_target=config.sigma_target,
                       dynamic_gm=config.dynamic_gm,
                       send_obs=config.send_obs,
                       auto_minimize=config.auto_minimize,
                       streamer_mode=config.streamer_mode,
                       send_gamestates=config.send_gamestates,
                       force_paging=config.force_paging,
                       local_cache_name="artemis_model_db").run()


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
            action_parser=match_config.action_parser
        ), match_config

    all_instances = {
        "aerial": 1,
        "recovery": 1,
        "default": 1
    }

    for index, data in enumerate(zip(all_instances.keys(), all_instances.values())):
        name, nb_instances = data
        if name not in version_dict:
            warnings.warn(f"Version \"{name}\" doesn't exist, create it or use the already existing keys in config.py : "
                          f"{list(version_dict.keys())}.\n"
                          f"Setting version to default", category=SyntaxWarning)
            name = "default"

        match, config = create_match(name)

        for i in range(nb_instances):
            process_name = f"{name.capitalize()}-{i + 1}"

            print(f"Starting process {process_name}")
            Process(target=target, args=(match, config), name=process_name).start()
