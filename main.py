import random

import numpy as np
from rlgym_sim.envs import Match
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from StateSetters import ProbabilisticStateSetter
from AstraObs import AstraObs
from sb3_multi_inst_env import SB3MultipleInstanceEnv


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 6
    num_instances = 1
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
    batch_size = target_steps // 10  # getting the batch size down to something more manageable - 100k in this case
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000


    def exit_save(model):
        model.save("models/exit_save")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=3,
            reward_function=
            EventReward(
                team_goal=50.0,
                concede=-100.0,
                goal=100),
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
            obs_builder=AstraObs(),
            state_setter=ProbabilisticStateSetter(),
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )


    env = SB3MultipleInstanceEnv(get_match, num_instances)
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="cuda",
            custom_objects={"n_envs": env.num_envs},
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
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            # may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback,
                        reset_num_timesteps=False)  # can ignore callback if training_interval < callback target
            model.save("models/exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
