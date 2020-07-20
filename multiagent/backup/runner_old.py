import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

# from baselines.common.runners import AbstractEnvRunner

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

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
    def __init__(self, env, world, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # World
        self.world = world

    def run(self):
        mb_eprew = []
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [self.obs],[],[],[],[]
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values = self.model.step(self.obs, self.nenv)
            # print(actions, values, actions.shape, values.shape)
            # input()
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            mb_obs.append(deepcopy(self.obs))
            mb_rewards.append(deepcopy(rewards))
            mb_actions.append(deepcopy(actions))
            mb_values.append(deepcopy(values))
            mb_dones.append(deepcopy(self.dones))
            if self.nenv == 1 and any(self.dones): self.obs[:] = self.env.reset()

        if self.nenv > 1:
            mb_obs = np.asarray(mb_obs[:-1], dtype=self.obs.dtype)
            mb_actions = np.asarray(mb_actions)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)

            last_values = self.model.value(self.obs, self.nenv)
            # discount/bootstrap off value fn
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                for i in range(self.world.n):
                    if t == self.nsteps - 1:
                        nextnonterminal = 1.0 - np.asarray(self.dones)
                        nextvalues = last_values[:,i]
                    else:
                        nextnonterminal = 1.0 - mb_dones[t+1]
                        nextvalues = mb_values[t+1][:,i]
                    delta = mb_rewards[t][:,i] + self.gamma * nextvalues * nextnonterminal - mb_values[t][:,i]
                    mb_advs[t][:,i] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
                    mb_returns[t][:,i] = mb_advs[t][:,i] + mb_values[t][:,i]
        else:
            mb_obs = np.asarray(mb_obs[:-1])#[:,np.newaxis,:]
            mb_actions = np.asarray(np.asarray(mb_actions))#[:,np.newaxis,:]
            mb_rewards = np.asarray(np.asarray(mb_rewards, dtype=np.float32))#[:,np.newaxis,:]
            mb_values = np.asarray(np.asarray(mb_values, dtype=np.float32))
            mb_dones = np.asarray(np.asarray(mb_dones, dtype=np.bool))#[:,np.newaxis]

            last_values = self.model.value(self.obs, self.nenv)
            print('ob0', mb_obs[:2], mb_obs.shape)
            print('rew0', mb_rewards[:2], mb_rewards.shape)
            print('values0', mb_values[:2], mb_values.shape)
            print('done0', mb_dones[:2], mb_dones.shape)
            print('last value', last_values.shape)
            input()

            # discount/bootstrap off value fn
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.nsteps)):
                for i in range(self.world.n):
                    if t == self.nsteps - 1:
                        nextnonterminal = 1.0 - np.asarray(self.dones)
                        nextvalues = last_values[:,i]
                    else:
                        nextnonterminal = 1.0 - mb_dones[t+1]
                        nextvalues = mb_values[t+1][:,i]
                    delta = mb_rewards[t][i] + self.gamma * nextvalues * nextnonterminal - mb_values[t][i]
                    mb_advs[t][:,i] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
                    mb_returns[t][:,i] = mb_advs[t][:,i] + mb_values[t][:,i]

            print('ob1', mb_obs[:2], mb_obs.shape)
            print('rew1', mb_rewards[:2], mb_rewards.shape)
            print('adv1', mb_advs[:2], mb_advs.shape)
            print('ret1', mb_returns[:2], mb_returns.shape)
            input()

            print('ob01', sf01(mb_obs)[:2], sf01(mb_obs).shape)
            print('rew01', sf01(mb_rewards)[:2], sf01(mb_rewards).shape)
            print('adv01', sf01(mb_advs)[:2], sf01(mb_advs).shape)
            print('ret01', sf01(mb_returns)[:2], sf01(mb_returns).shape)
            input()

        mb_obs = np.concatenate(mb_obs, axis=0).transpose()
        mb_actions = np.concatenate(mb_actions, axis=0).transpose()
        mb_rewards = np.concatenate(mb_rewards, axis=0).transpose()
        mb_values = np.concatenate(mb_values, axis=0).transpose()
        mb_advs = np.concatenate(mb_advs, axis=0).transpose()
        mb_returns = np.concatenate(mb_returns, axis=0).transpose()
        mb_dones = np.concatenate(mb_dones, axis=0).transpose()

        # print('ob2', mb_obs[:2], mb_obs.shape)
        # print('rew2', mb_rewards[:2], mb_rewards.shape)
        # print('adv2', mb_advs[:2], mb_advs.shape)
        # print('ret2', mb_returns[:2], mb_returns.shape)
        # input()

        mb_obs = [np.vstack(obs_n) for obs_n in mb_obs]
        mb_actions = [np.vstack(ac_n) for ac_n in mb_actions]
        mb_rewards = [np.vstack(rw_n) for rw_n in mb_rewards]
        mb_values = [np.vstack(vl_n) for vl_n in mb_values]
        mb_advs = [np.vstack(ad_n) for ad_n in mb_advs]
        mb_returns = [np.vstack(rt_n) for rt_n in mb_returns]

        # print('ob3', mb_obs)
        # print('rew3', mb_rewards[1])
        # assert(all(mb_rewards[1]==0))
        # print('adv3', mb_advs)
        # print('ret3', mb_returns)
        # input()

        # share actions
        mb_actions = self.model.share_actions(mb_actions)

        return mb_obs, mb_actions, mb_rewards, mb_values, mb_advs, mb_returns, mb_dones

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


