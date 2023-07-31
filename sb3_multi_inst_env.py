import multiprocessing as mp
import os
import time
from typing import Optional, List, Union, Any, Callable, Sequence, Iterable

import numpy as np

from Envs import ObservableGym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvIndices,
)
from stable_baselines3.common.vec_env.subproc_vec_env import _worker

from rlgym_sim.envs import Match
from rlgym_sim.gym import Gym


class SB3MultipleInstanceEnv(SubprocVecEnv):
    """
    Class for launching several Rocket League instances into a single SubprocVecEnv for use with Stable Baselines.
    """

    MEM_INSTANCE_LAUNCH = 3.5e9
    MEM_INSTANCE_LIM = 4e6

    @staticmethod
    def estimate_supported_processes():
        import psutil

        vm = psutil.virtual_memory()
        # Need 3.5GB to launch, reduces to 350MB after a while
        est_proc_mem = round(
            (vm.available - SB3MultipleInstanceEnv.MEM_INSTANCE_LAUNCH)
            / SB3MultipleInstanceEnv.MEM_INSTANCE_LAUNCH
        )
        est_proc_cpu = os.cpu_count()
        est_proc = min(est_proc_mem, est_proc_cpu)
        return est_proc

    def __init__(
            self,
            match_func_or_matches: Union[Callable[[], Match], Sequence[Match]],
            num_instances: Optional[int] = None,
            wait_time: float = 0,
            tick_skip: float = 8,
            dodge_deadzone: float = 0.5,
            copy_gamestate_every_step: bool = False
    ):
        """
        :param match_func_or_matches: either a function which produces the a Match object, or a list of Match objects.
                                Needs to be a function so that each subprocess can call it and get their own objects.
        :param num_instances: the number of Rocket League instances to start up,
                              or "auto" to estimate how many instances are supported (requires psutil).
        :param wait_time: the time to wait between launching each instance. Default one minute.
        :param force_paging: enable forced paging of each spawned rocket league instance to reduce memory utilization
                             immediately, instead of allowing the OS to slowly page untouched allocations.
                             WARNING: This will require you to potentially expand your Windows Page File, and it may
                             substantially increase disk activity, leading to decreased disk lifetime.
                             Use at your own peril.
                             https://www.tomshardware.com/news/how-to-manage-virtual-memory-pagefile-windows-10,36929.html
                             Default is off: OS dictates the behavior.
        """
        if callable(match_func_or_matches):
            assert num_instances is not None, (
                "If using a function to generate Match objects, "
                "num_instances must be specified"
            )
            if num_instances == "auto":
                num_instances = SB3MultipleInstanceEnv.estimate_supported_processes()
            match_func_or_matches = [
                match_func_or_matches() for _ in range(num_instances)
            ]

        self.all_matches = match_func_or_matches

        def get_process_func(i):
            def spawn_process():
                match = match_func_or_matches[i]
                env = ObservableGym(
                    match,
                    copy_gamestate_every_step=copy_gamestate_every_step,
                    dodge_deadzone=dodge_deadzone,
                    tick_skip=tick_skip,
                    gravity=1,
                    boost_consumption=1
                )
                return env

            return spawn_process

        # super().__init__([])  Super init intentionally left out since we need to launch processes with delay

        env_fns = [get_process_func(i) for i in range(len(match_func_or_matches))]

        # START - Code from SubprocVecEnv class
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)



        # Fork is not a thread safe method (see issue #217)
        # but is more user friendly (does not require to wrap the code in
        # a `if __name__ == "__main__":`)
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for index, data in enumerate(zip(
                self.work_remotes, self.remotes, env_fns
        )):
            work_remote, remote, env_fn = data
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            print(f"Instance {index + 1} started")
            self.processes.append(process)
            work_remote.close()

            if len(self.processes) != len(env_fns):
                time.sleep(wait_time)  # ADDED - Waits between starting Rocket League instances

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        # END - Code from SubprocVecEnv class

        self.n_agents_per_env = [m.agents for m in match_func_or_matches]
        self.max_agents_per_env = [m.team_size * 2 if m.spawn_opponents else m.team_size for m in match_func_or_matches]

        self.num_envs = sum(self.n_agents_per_env)
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def pre_reset(self):
        for remote in self.remotes:
            remote.send(("pre_reset", None))

    def reset(self) -> VecEnvObs:
        self.n_agents_per_env = self.get_attr(("_match", "agents"))
        for remote in self.remotes:
            remote.send(("reset", None))

        flat_obs = np.zeros((sum(self.max_agents_per_env), *self.observation_space.shape))
        current_pos = 0
        for remote, n_agents, max_agents in zip(self.remotes, self.n_agents_per_env, self.max_agents_per_env):
            obs = remote.recv()
            padding_amount = max_agents - n_agents

            flat_obs[current_pos:current_pos + n_agents] = np.asarray(obs)
            flat_obs[current_pos + n_agents:current_pos + max_agents] = np.zeros(
                (padding_amount, *self.observation_space.shape))

            current_pos += max_agents

        return np.asarray(flat_obs)

    def get_attr(self, attr_name: Union[str, Iterable[str]], indices: VecEnvIndices = None) -> List[Any]:
        for remote in self.remotes:
            remote.send(("get_attr", attr_name))

        attrs = []
        for remote in self.remotes:
            attr = remote.recv()
            attrs.append(attr)

        return attrs

    def step_async(self, actions: np.ndarray) -> None:
        self.n_agents_per_env = self.get_attr(("_match", "agents"))

        i = 0
        for remote, n_agents in zip(self.remotes, self.n_agents_per_env):
            remote.send(("step", actions[i: i + n_agents]))
            i += n_agents
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        flat_obs = np.zeros((sum(self.max_agents_per_env), *self.observation_space.shape))
        flat_rewards = np.zeros((sum(self.max_agents_per_env), ))
        flat_done = []
        flat_info = []
        current_pos = 0

        for remote, n_agents, max_agents in zip(self.remotes, self.n_agents_per_env, self.max_agents_per_env):
            obs, rew, done, info = remote.recv()
            padding_amount = max_agents - n_agents

            flat_obs[current_pos:current_pos + n_agents] = np.asarray(obs)
            flat_rewards[current_pos:current_pos + n_agents] = np.array(rew)
            flat_done += [done] * max_agents
            flat_info += [info] * max_agents

            flat_obs[current_pos + n_agents:current_pos + max_agents] = np.zeros((padding_amount, *self.observation_space.shape))
            flat_rewards[current_pos + n_agents:current_pos + max_agents] = np.zeros((padding_amount, ))

            current_pos += max_agents

        return flat_obs, flat_rewards, np.array(flat_done), flat_info


    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        res = super(SB3MultipleInstanceEnv, self).seed(seed)
        return sum([r] * a for r, a in zip(res, self.n_agents_per_env))

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        # Override to prevent out of bounds
        indices = self._get_indices(indices)
        remotes = []
        for i in indices:
            tot = 0
            for remote, n_agents in zip(self.remotes, self.n_agents_per_env):
                tot += n_agents
                if i < tot:
                    remotes.append(remote)
                    break
        return remotes
