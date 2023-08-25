import re

import numpy as np
import rlgym_ppo
from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import AnyCondition, TimeoutCondition, GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions.combined_reward import CombinedReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

from Rewards import DistancePlayerToBall, SimToGymWrapper
from StateSetters import ProbabilisticStateMutator, MutatorProb, DynamicScoredReplayMutator, AerialStateMutator, \
    SaveMutator, ShotMutator, SideHighRollMutator, ShortGoalRollMutator, AirDribble2TouchMutator, AirDribbleSetupMutator
from VoidParser import VoidParser

frame_skip = 8  # Number of ticks to repeat an action
half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5

fps = 120 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
agents_per_match = 6
target_steps = 10_000

num_instances = 3

steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
batch_size = target_steps // 10  # getting the batch size down to something more manageable - 100k in this case
training_interval = 20_000
mmr_save_frequency = 50_000_000


def create_match():
    return RLGym(
        state_mutator=ProbabilisticStateMutator(
            # Mandatory mutators (will be replaced by dynamic)
            (
                FixedTeamSizeMutator(),
            ),

            # The rest
            MutatorProb(mutator=KickoffMutator(), probability=50),
            MutatorProb(mutator=ShotMutator(), probability=40),
            MutatorProb(mutator=SaveMutator(), probability=20),
            MutatorProb(mutator=SideHighRollMutator(), probability=25),
            MutatorProb(mutator=ShortGoalRollMutator(), probability=10),
            MutatorProb(mutator=AerialStateMutator(), probability=40),
            MutatorProb(mutator=AirDribble2TouchMutator(), probability=30),
            MutatorProb(mutator=AirDribbleSetupMutator(), probability=40),
        ),
        action_parser=VoidParser(),
        reward_fn=CombinedReward(
            (SimToGymWrapper(DistancePlayerToBall()), 0.1)
        ),
        obs_builder=DefaultObs(zero_padding=3),

        # Conditions which infers that the objective hasn't been reached
        truncation_cond=AnyCondition(
            TimeoutCondition(fps * 300)
        ),

        # Conditions which infers that the objective has been reached
        termination_cond=AnyCondition(
            GoalCondition()
        ),

        # Unless we use rlviser, useless
        renderer=None,

        # Just to say "we use rlgym_sim"
        transition_engine=RocketSimEngine()
    )


if __name__ == "__main__":

    model = rlgym_ppo.Learner(
        # Env configuration
        env_create_function=create_match,
        n_proc=num_instances,

        # PPO Configuration
        ts_per_iteration=steps,
        ppo_epochs=10,
        ppo_batch_size=batch_size,
        policy_layer_sizes=(512, 512, 512, 512),
        critic_layer_sizes=(512, 512, 512),
        policy_lr=5e-5,
        critic_lr=5e-5,
        exp_buffer_size=target_steps,
        device="cpu",
        gae_gamma=gamma,
        gae_lambda=0.95,
        ppo_ent_coef=0.01,

        # Save/Load frequency
        save_every_ts=mmr_save_frequency,
        checkpoints_save_folder="models_checkpoints/model-",
        checkpoint_load_folder="models_checkpoints/model-",

        # Wandb
        load_wandb=True,
        log_to_wandb=True,
        wandb_run_name="Void",
        wandb_group_name="madaos",
        wandb_project_name="Void",
    )

    try:
        model.load("exit_save", load_wandb=True)
        print("Model loaded")
    except Exception as e:
        print("Got exception : " + str(e))
        print("Model not found, creating a new one...")


    def rewards_to_text(rewards):
        return list([re.sub(r"(\w)([A-Z])", r"\1 \2", reward.__class__.__name__) for reward in rewards])


    try:
        mmr_model_target_count = model.agent.cumulative_timesteps + mmr_save_frequency
        while True:
            model.learn()  # can ignore callback if training_interval < callback target
            # model.save(f"models/{Worker.current_model}/exit_save")
            if model.agent.cumulative_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.agent.cumulative_timesteps}")
                mmr_model_target_count += mmr_save_frequency
    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    model.save(model.agent.cumulative_timesteps)
    print("Save complete")
