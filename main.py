import re

import numpy as np
from rlgym_sim.envs import Match
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from StateSetters import ProbabilisticStateSetter
from sb3_multi_inst_env import SB3MultipleInstanceEnv

frame_skip = 8  # Number of ticks to repeat an action
half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5

fps = 120 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
agents_per_match = 1
target_steps = 100_000

num_instances = 3

steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
batch_size = steps * num_instances * agents_per_match * 2  # getting the batch size down to something more manageable - 100k in this case
training_interval = 20_000
mmr_save_frequency = 50_000_000
exit_if_load_fails = True

if __name__ == "__main__":

    from config import version_dict, Configuration


    def create_match():

        match_config: Configuration = version_dict["default"]

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
        )


    env = SB3MultipleInstanceEnv(create_match,
                                 num_instances=num_instances)
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards


    def exit_save(model):
        model.save("models/exit_save")

    try:
        model = PPO.load(
            "models/exit_save",
            env,
            device="cuda",
            custom_objects={
                "num_envs": env.num_envs,
                "n_steps": steps,
                "batch_size": batch_size,
            },
        )
        print("Loaded previous exit save.")

        params = ["num_timesteps", "batch_size", "action_space", "n_steps", "n_envs", "device"]
        biggest_str = len(max(params, key=lambda x: len(x)))

        print("Model status: ")

        for attr in params:
            print(f"\t{attr: <{biggest_str + 4}} {model.__dict__[attr]:{'_' if isinstance(model.__dict__[attr], (int, float)) else ''}}")


    except Exception as e:
        print(f"No saved model found due to {e.__class__.__name__}. (Message : {str(e)})")
        if exit_if_load_fails:
            print("You set the \"exit_if_load_fails\" attribute to True, loading failed, exiting")
            exit(1)
        print("Creating a new model")
        from torch.nn import LeakyReLU

        policy_kwargs = dict(
            activation_fn=LeakyReLU,
            net_arch=[dict(
                pi=[512, 512, 512, 512],
                vf=[512, 512, 512])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,  # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,  # Around this is fairly common for PPO
            ent_coef=0.01,  # From PPO Atari
            vf_coef=1.,  # From PPO Atari
            gamma=gamma,  # Gamma as calculated using half-life
            verbose=3,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_steps=steps,  # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="cuda"  # Uses GPU if available
        )

        # Save model every so often
        # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
        # This saves to specified folder with a specified name


    def rewards_to_text(rewards):
        return list([re.sub(r"(\w)([A-Z])", r"\1 \2", reward.__class__.__name__) for reward in rewards])

    reward_legends = rewards_to_text(version_dict["default"].rewards[0])

    # may need to reset timesteps when you're running a different number of instances than when you saved the model
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models",
                                  name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(training_interval,
                        callback=[callback, SB3CombinedLogRewardCallback(reward_names=reward_legends)],
                        reset_num_timesteps=False)
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency
    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
