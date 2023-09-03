import numpy as np
import rlgym_ppo
import rlgym_sim.make
from rlgym_sim.utils.reward_functions import CombinedReward

from StateSetters import ProbabilisticStateSetter, TeamSizeSetter

frame_skip = 8  # Number of ticks to repeat an action
half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5

fps = 120 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
agents_per_match = 1
target_steps = 100_000

num_instances = 1

steps = target_steps // (num_instances * agents_per_match)  # making sure the experience counts line up properly
batch_size = steps * num_instances * agents_per_match * 2  # getting the batch size down to something more manageable - 100k in this case
training_interval = 20_000
mmr_save_frequency = 50_000_000
exit_if_load_fails = True


def create_match():
    from config import version_dict, Configuration
    match_config: Configuration = version_dict["default"]

    return rlgym_sim.make(
        team_size=match_config.team_size,
        reward_fn=CombinedReward(
            reward_functions=match_config.rewards[0],
            reward_weights=match_config.rewards[1]
        ),
        spawn_opponents=match_config.spawn_opponents,
        terminal_conditions=match_config.terminal_conditions,
        obs_builder=match_config.obs_builder,
        state_setter=TeamSizeSetter(),
        action_parser=match_config.action_parser,

        tick_skip=8,
        gravity=1,
        copy_gamestate_every_step=False,
        boost_consumption=1,
        dodge_deadzone=0.8
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
        checkpoints_save_folder="models_checkpoints/model",
        # checkpoint_load_folder="models_checkpoints/model", # Uncomment this when one model already exists, crashes when there are no models

        # Wandb
        load_wandb=True,
        log_to_wandb=True,
        wandb_run_name="Void",
        wandb_group_name="madaos",
        wandb_project_name="Void",
    )

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
