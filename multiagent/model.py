import tensorflow as tf
import numpy as np
import time, copy

class Model(object):
    def __init__(self, env, world, policies, admm_iter, mode, ob_normalization):
        self.env = env
        self.world = world
        self.policies = policies
        self.admm_iter = admm_iter 
        self.mode = mode
        self.leader = 0 if mode == 'central' else None
        self.test = False
        self.ob_normalization = ob_normalization
        for i in range(len(world.agents)):
            policies[i].vfadam.sync()

    def step(self, obs):
        action_n, value_n, neglogp_n = [], [], []
        if not self.test:
            for i in range(len(self.world.agents)):
                action, value, _, neglogp = self.policies[i].pi.step(np.stack(obs[:,i], axis=0))
                if self.mode=='matrpo':
                    if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                    action_n.append(tf.gather(action,i,axis=1).numpy())
                elif self.mode=='trpo':
                    action_n.append(action.numpy())
                elif self.mode == 'central':
                    if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                    if i == self.leader: action_n += list(action.numpy().transpose())
                value_n.append(value.numpy())
                neglogp_n.append(neglogp.numpy())

            actions = np.vstack(action_n).transpose()
            values = np.vstack(value_n).transpose()
            neglogps = np.vstack(neglogp_n).transpose()

            return actions, values, neglogps
        else:
            for i in range(len(self.world.agents)):
                action, value, _, neglogp = self.policies[i].pi.step(np.expand_dims(obs[i], axis=0))
                if self.mode=='matrpo':
                    if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                    action_n.append(action[0,i].numpy())
                elif self.mode=='trpo':
                    action_n.append(action.numpy())
                elif self.mode=='central':
                    if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                    if i == self.leader: action_n += list(action[0].numpy())
                value_n.append(value)
                neglogp_n.append(neglogp)

            return action_n, value_n, neglogp_n

    def value(self, obs):
        return np.vstack([self.policies[i].pi.value(tf.stack(obs[:,i], axis=0)) 
                          for i in range(len(self.world.agents))]).transpose()

    def share_actions(self, actions):
        shared_actions_n = []
        for i in range(len(self.world.agents)):
            if self.mode == 'matrpo':
                shared_actions_n.append(actions)
            elif self.mode == 'trpo':
                shared_actions_n.append(actions[:,i:i+1])
            elif self.mode == 'central':
                shared_actions_n.append(actions)

        return shared_actions_n

    def save(self):
        for pi in self.policies:
            save_path = pi.manager.save()
            print("Save checkpoint to {}".format(save_path))

    def train(self, actions, obs, returns, dones, values, advs, neglogpacs):
        eps = 1e-8
        A = self.world.comm_matrix
        edges = A[np.unique(np.nonzero(A)[0])]
        # Policy Update
        if self.mode == 'matrpo':
            for i in range(len(self.world.agents)):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                self.policies[i].reinitial_estimates()
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
            # prepare data
            norm_advs = [(adv-np.mean(advs))/(np.std(advs)+eps) for adv in advs]
            argvs = tuple(zip(obs, actions, norm_advs, returns, values))
            # consensus using admm
            for itr in range(self.admm_iter):
                # edge = edges[np.random.choice(range(len(adv_edges)))]
                edge = edges[itr % len(edges)]
                q = np.where(edge != 0)[0]
                k, j = q[0], q[-1]
                # Update Agent k and j
                self.policies[k].update(*argvs[k])
                self.policies[j].update(*argvs[j])
                ratio_k, multipliers_k = self.policies[k].info_to_exchange(obs[k], actions[k], j)
                ratio_j, multipliers_j = self.policies[j].info_to_exchange(obs[j], actions[j], k)
                self.policies[k].exchange(obs[k], actions[k], edge[k], ratio_j, multipliers_j, j)
                self.policies[j].exchange(obs[j], actions[j], edge[j], ratio_k, multipliers_k, k)
        elif self.mode == 'central':
            if self.ob_normalization:
                self.policies[self.leader].pi.ob_rms.update(obs[self.leader])
                self.policies[self.leader].oldpi.ob_rms.update(obs[self.leader])
            norm_advs = (advs[self.leader] - np.mean(advs[self.leader])) / (np.std(advs[self.leader])+eps)
            argvs = (obs[self.leader], actions[self.leader], norm_advs, returns[self.leader], values[self.leader])
            self.policies[self.leader].assign_old_eq_new()
            self.policies[self.leader].vfupdate(obs[self.leader], returns[self.leader], values[self.leader])
            self.policies[self.leader].trpo_update(*argvs)
        else:
            norm_advs = copy.deepcopy(advs)
            for i in range(len(self.world.agents)):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                norm_advs[i] = (advs[i]-np.mean(advs[i]))/(np.std(advs[i])+eps)
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
                self.policies[i].trpo_update(obs[i], actions[i], norm_advs[i], returns[i], values[i])
