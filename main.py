import re
import warnings
from collections import Counter
from multiprocessing import Process

import numpy as np
from rlgym_sim.envs import Match
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from MyPPO import MyPPO
from StateSetters import ProbabilisticStateSetter
from config import version_dict, Configuration
from match import DynamicGMMatch
from sb3_multi_inst_env import SB3MultipleInstanceEnv

class Worker:
    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 6
    num_instances = 1
    target_steps = 10_000
    steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
    batch_size = target_steps // 10  # getting the batch size down to something more manageable - 100k in this case
    training_interval = 20_000
    mmr_save_frequency = 50_000_000

    current_model = "default"

    def __init__(self, pipe):
        self.pipe = pipe

        self.all_configs = [Worker.current_model]

        self.env = SB3MultipleInstanceEnv([self.create_match(Worker.current_model)] * Worker.num_instances,
                                          num_instances=Worker.num_instances)
        self.env = VecCheckNan(self.env)  # Optional
        self.env = VecMonitor(self.env)  # Recommended, logs mean reward and ep_len to Tensorboard
        self.env = VecNormalize(self.env, norm_obs=False, gamma=Worker.gamma)  # Highly recommended, normalizes rewards

    def exit_save(self, model):
        model.save(f"models/exit_save")

    def create_match(self, version):

        if version not in version_dict:
            warnings.warn(f"Version '{version}' is not in the config dictionary, "
                          f"possible keys are {list(version_dict.keys())}, switching version to default")
            version = "default"

        match_config: Configuration = version_dict[version]

        return DynamicGMMatch(
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
            action_parser=match_config.action_parser,
            gm_weights=[0.1, 0.8, 0.1]
        )

    def run(self):
        try:
            model = MyPPO.load(
                f"models/{Worker.current_model}/exit_save.zip",
                self.env,
                device="cuda",
                custom_objects={"num_envs": self.env.num_envs}
            )
            print("Loaded previous exit save.")
        except:
            print("No saved model found, creating new model.")
            from torch.nn import LeakyReLU

            policy_kwargs = dict(
                activation_fn=LeakyReLU,
                net_arch=[dict(
                    pi=[512, 512, 512, 512],
                    vf=[512, 512, 512])],
            )

            model = MyPPO(
                MlpPolicy,
                self.env,
                n_epochs=10,  # PPO calls for multiple epochs
                policy_kwargs=policy_kwargs,
                learning_rate=5e-5,  # Around this is fairly common for PPO
                ent_coef=0.01,  # From PPO Atari
                vf_coef=1.,  # From PPO Atari
                gamma=Worker.gamma,  # Gamma as calculated using half-life
                verbose=3,  # Print out all the info as we're going
                batch_size=Worker.batch_size,  # Batch size as high as possible within reason
                n_steps=Worker.steps,  # Number of steps to perform before optimizing network
                tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
                device="cuda"  # Uses GPU if available
            )

        # Save model every so often
        # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
        # This saves to specified folder with a specified name

        def rewards_to_text(rewards):
            return list([re.sub(r"(\w)([A-Z])", r"\1 \2", reward.__class__.__name__) for reward in rewards])

        all_rewards = []
        for c in self.all_configs:
            if c not in version_dict:
                c = "default"

            all_rewards.extend(version_dict[c].rewards[0])

        all_rewards = tuple([i for i, c in Counter(all_rewards).items() if c == 1])

        reward_legends = rewards_to_text(all_rewards)

        # may need to reset timesteps when you're running a different number of instances than when you saved the model
        callback = CheckpointCallback(round(5_000_000 / self.env.num_envs), save_path=Worker.current_model,
                                      name_prefix=f"rl_model_{Worker.current_model}")

        try:
            mmr_model_target_count = model.num_timesteps + Worker.mmr_save_frequency
            while True:
                model.learn(Worker.training_interval,
                            callback=[callback, SB3CombinedLogRewardCallback(reward_names=reward_legends)],
                            reset_num_timesteps=False)  # can ignore callback if training_interval < callback target
                # model.save(f"models/{Worker.current_model}/exit_save")
                if model.num_timesteps >= mmr_model_target_count:
                    model.save(f"mmr_models/{model.num_timesteps}")
                    mmr_model_target_count += Worker.mmr_save_frequency
        except KeyboardInterrupt:
            print("Exiting training")

        print("Saving model")
        self.exit_save(model)
        print("Save complete")


if __name__ == "__main__":
    w = Worker(None).run()
