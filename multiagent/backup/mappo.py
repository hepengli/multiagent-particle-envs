import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam

import multiagent.scenarios as scenarios
from multiagent.model import Model
from multiagent.runner import Runner
from multiagent.build_policy import build_policy

import gym
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers, set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        # try:
        #     low, high = self.env.unwrapped._feasible_action()
        # except:
        #     low, high = self.action_space.low, self.action_space.high

        # import numpy as np
        # action = np.nan_to_num(action)
        # action = np.clip(action, low, high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class MAPPO(object):
    """ Paralell CPO algorithm """
    def __init__(self, env_id, nsteps, gamma=0.995, lam=0.95, ent_coef=0.0, admm_iter=10, rho=1.0, 
                 vf_coef=0.5, max_grad_norm=0.5, num_env=1, reward_scale=1.0, seed=None, load_path=None, 
                 logger_dir=None, force_dummy=False, info_keywords=(), **network_kwargs):
        # Setup stuff
        set_global_seeds(seed)
        np.set_printoptions(precision=5)

        # Scenario
        scenario = scenarios.load('{}.py'.format(env_id)).Scenario()
        world = scenario.make_world()

        if hasattr(world.agents[0], 'adversary'):
            good_agents = [agent for agent in world.agents if not agent.adversary]
            world.n_agt = len(good_agents)
            world.n_adv = len(world.agents) - world.n_agt
            world.n = world.n_agt+world.n_adv
        else:
            world.n_agt = 0
            world.n_adv = len(world.agents)
            world.n = world.n_agt+world.n_adv
        self.n_agt, self.n_adv, self.n = world.n_agt, world.n_adv, world.n

        # Environment
        if num_env == 1:
            env = self.make_env(env_id, seed, logger_dir=logger_dir, reward_scale=1.0, mpi_rank=0, subrank=0, info_keywords=info_keywords)
        else:
            env = self.make_vec_env(env_id, seed, logger_dir=logger_dir, num_env=num_env, reward_scale=reward_scale,
                                        force_dummy=force_dummy, info_keywords=info_keywords)
        self.env = env

        # create interactive policies for each agent
        self.policies = policies = [build_policy(env, world, i, network_kwargs['n_hiddens']) for i in range(len(world.agents))]
        # model
        self.model = model = Model(env=env, world=world, policies=policies, nsteps=nsteps, load_path=load_path, rho=rho, 
                                   admm_iter=admm_iter, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
        # runner
        self.runner = Runner(env=env, world=world, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    def make_env(self, env_id, seed, logger_dir=None, reward_scale=1.0, mpi_rank=0, subrank=0, info_keywords=()):
        """
        Create a wrapped, monitored gym.Env for safety.
        """
        scenario = scenarios.load('{}.py'.format(env_id)).Scenario()
        world = scenario.make_world()
        env_dict = {
            "world": world,
            'reset_callback': scenario.reset_world,
            'reward_callback': scenario.reward, 
            'observation_callback': scenario.observation,
            'info_callback': None,
            'done_callback': scenario.done, 
            'shared_viewer':  True
            }
        env = gym.make('MultiAgent-v0', **env_dict)
        env.seed(seed + subrank if seed is not None else None)
        env = Monitor(env,
                    logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                    allow_early_resets=True,
                    info_keywords=info_keywords)
        env = ClipActionsWrapper(env)
        if reward_scale != 1.0:
            from baselines.common.retro_wrappers import RewardScaler
            env = RewardScaler(env, reward_scale)
        return env

    def make_vec_env(self, env_id, seed, logger_dir=None, reward_scale=1.0, num_env=1, force_dummy=False, info_keywords=()):
        """
        Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
        """
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        seed = seed + 10000 * mpi_rank if seed is not None else None
        def make_thunk(rank, initializer=None):
            return lambda: self.make_env(
                env_id,
                seed,
                logger_dir=logger_dir,
                reward_scale=reward_scale,
                mpi_rank=mpi_rank,
                subrank=rank,
                info_keywords=info_keywords,
            )
        set_global_seeds(seed)

        if not force_dummy and num_env > 1:
            return SubprocVecEnv([make_thunk(i) for i in range(num_env)])
        else:
            return DummyVecEnv([make_thunk(i) for i in range(num_env)])
