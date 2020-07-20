import tensorflow as tf
import numpy as np
import time

class Model(object):
    def __init__(self, env, world, policies, ncommtime):
        self.env = env
        self.world = world
        self.policies = policies
        self.nsteps = policies[0].nsteps
        self.ncommtime = ncommtime
        for i in range(world.n):
            policies[i].vfadam.sync()


    def step(self, obs_n):
        action_n, value_n, neglogp_n = [], [], []
        for i in range(self.env.n):
            action, value, _, neglogp = self.policies[i].pi.step(obs_n[i])
            action_n.append(np.diag(action._numpy()))
            value_n.append(value)
            neglogp_n.append(neglogp)

        return action_n, value_n, neglogp_n

    def value(self, obs_n):
        return [self.policies[i].pi.value(obs_n[i]) for i in range(self.env.n)]

    def share_actions(self, actions_n):
        shared_actions_n = list(actions_n)
        if hasattr(self.env.agents[0], 'adversary'):
            n_a = self.env.world.num_adversaries
            n_g = self.env.world.num_good_agents
        else:
            n_a = len(self.env.agents)
            n_g= 0

        for i in range(n_a):
            shared_actions_n[i] = np.hstack(actions_n[:n_a])
        for i in range(n_g):
            shared_actions_n[n_a+i] = np.hstack(actions_n[n_a:]).astype(actions_n[n_a+i].dtype).squeeze()

        return shared_actions_n

    def train(self, obs, returns, dones, actions, values, atarg, neglogpacs):
        for i in range(self.env.n):
            if hasattr(self.policies[i].pi, "ret_rms"): self.policies[i].pi.ret_rms.update(returns)
            if hasattr(self.policies[i].pi, "ob_rms"): self.policies[i].pi.ob_rms.update(obs) # update running mean/std for policy
            # hvp.append(lambda p: self.allmean(self.policies[i].compute_hvp(p, obs[i], actions[i]).numpy()) + cg_damping * p)
            self.policies[i].reinitial_estimates()
            self.policies[i].assign_old_eq_new() # set old parameter values to new parameter values
            atarg[i] = (atarg[i] - np.mean(atarg)) / np.std(atarg) # standardized advantage function estimate

        argvs = list(zip(obs, actions, atarg, returns, values))
        # some constants
        eps = 1e-8
        A = self.world.comm_matrix
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]
        for itr in range(self.ncommtime):
            edge = adv_edges[np.random.choice(range(len(adv_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]
            self.policies[k].assign_new_eq_old()
            self.policies[j].assign_new_eq_old()
            # Update Agent k
            self.policies[k].update(*argvs[k], j)
            self.policies[j].update(*argvs[j], k)
            # input()
            ratio_k, multipliers_k = self.policies[k].info_to_exchange(obs[k], actions[k], j)
            ratio_j, multipliers_j = self.policies[j].info_to_exchange(obs[j], actions[j], k)
            self.policies[k].exchange(obs[k], actions[k], ratio_j, multipliers_j, j)
            self.policies[j].exchange(obs[j], actions[j], ratio_k, multipliers_k, k)

            print('----------------------------')
            print('Agent 0 estimates: \n', self.policies[0].estimates[1])
            print('Agent 1 estimates: \n', self.policies[1].estimates[0])
            print('----------------------------')
            print('Agent 0 ratios: \n', ratio_k.numpy())
            print('Agent 1 ratios: \n', ratio_j.numpy())
            # input()
