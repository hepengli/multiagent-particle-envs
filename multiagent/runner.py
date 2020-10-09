import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
import tensorflow as tf
from baselines.common.vec_env.vec_env import VecEnv

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps, finite):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.finite = finite

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env, world, model, nsteps, gamma, lam, finite):
        super().__init__(env=env, model=model, nsteps=nsteps, finite=finite)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # World
        self.world = world

    def run(self):
        monitor_rewards = []
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones, mb_neglogpacs = [self.obs],[],[],[],[],[]
        # For n in range number of steps
        eps_rewards = []
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, neglogpacs = self.model.step(self.obs)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)

            mb_obs.append(deepcopy(self.obs))
            mb_actions.append(deepcopy(actions))
            mb_rewards.append(deepcopy(rewards))
            mb_values.append(deepcopy(values))
            mb_neglogpacs.append(deepcopy(neglogpacs))
            mb_dones.append(deepcopy(self.dones))
            eps_rewards.append(deepcopy(rewards))

        monitor_rewards = np.asarray(monitor_rewards, dtype=np.float32)
        mb_obs = np.asarray(mb_obs[:-1])
        mb_actions = np.asarray(mb_actions)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        for i in range(len(self.world.agents)):
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones * self.finite
                    nextvalues = last_values[:,i]
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1] * self.finite
                    nextvalues = mb_values[t+1,:,i]
                delta = mb_rewards[t,:,i] + self.gamma * nextvalues * nextnonterminal - mb_values[t,:,i]
                mb_advs[t,:,i] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (self.model.share_actions(sf01(mb_actions)), sf03(sf(mb_obs)), 
                *map(sf, (mb_rewards, mb_returns, mb_dones, mb_values, mb_advs, mb_neglogpacs)))


def sf(arr):
    return sf02(sf01(arr))

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def sf02(arr):
    """
    swap axes 0 and 1
    """
    s = arr.shape
    if len(s) > 1:
        return list(arr.swapaxes(0, 1))
    else:
        return arr

def sf03(obs_n):
    return [np.vstack(obs) for obs in obs_n]
