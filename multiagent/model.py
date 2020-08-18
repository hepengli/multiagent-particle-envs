import tensorflow as tf
import numpy as np
import time

class Model(object):
    def __init__(self, env, world, policies, admm_iter, adv, agt, ob_normalization):
        self.env = env
        self.world = world
        self.policies = policies
        self.nsteps = policies[0].nsteps
        if isinstance(admm_iter, list):
            assert len(admm_iter) == 2
            self.admm_iter = admm_iter 
        else:
            self.admm_iter = [admm_iter]*2
        self.adv = adv
        self.agt = agt
        self.ob_normalization = ob_normalization
        for i in range(world.n):
            policies[i].vfadam.sync()

    def step(self, obs):
        action_n, value_n, neglogp_n = [], [], []
        if not isinstance(obs, list):
            for i in range(self.world.n):
                action, value, _, neglogp = self.policies[i].pi.step(np.stack(obs[:,i], axis=0))
                if i < self.world.n_adv:
                    if self.adv == 'centralized':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        if i == 0: action_n += list(action.numpy().transpose())
                    if self.adv=='cooperative':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        action_n.append(tf.gather(action,i,axis=1).numpy())
                    if self.adv=='independent':
                        action_n.append(action.numpy())
                else:
                    if self.agt == 'centralized':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        if i == self.world.n_adv: action_n += list(action.numpy().transpose())
                    if self.agt=='cooperative':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        action_n.append(tf.gather(action,i-self.world.n_adv,axis=1).numpy())
                    if self.agt=='independent':
                        action_n.append(action.numpy())
                value_n.append(value.numpy())
                neglogp_n.append(neglogp.numpy())

            actions = np.vstack(action_n).transpose()
            values = np.vstack(value_n).transpose()
            neglogps = np.vstack(neglogp_n).transpose()

            return actions, values, neglogps
        else:
            for i in range(self.world.n):
                action, value, _, neglogp = self.policies[i].pi.step(np.expand_dims(obs[i], axis=0))
                if i < self.world.n_adv:
                    if self.adv=='centralized':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        if i == 0: action_n += list(action[0].numpy())
                    if self.adv=='cooperative':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        action_n.append(action[0,i].numpy())
                    if self.adv=='independent':
                        action_n.append(action.numpy())
                else:
                    if self.agt=='centralized':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        if i == self.world.n_adv: action_n += list(action[0].numpy())
                    if self.agt=='cooperative':
                        if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                        action_n.append(action[0,i-self.world.n_adv].numpy())
                    if self.agt=='independent':
                        action_n.append(action.numpy())
                value_n.append(value)
                neglogp_n.append(neglogp)

            return action_n, value_n, neglogp_n


    def value(self, obs):
        return np.vstack([self.policies[i].pi.value(tf.stack(obs[:,i], axis=0)) for i in range(self.world.n)]).transpose()


    def share_actions(self, actions):
        s = actions.shape
        assert s[0] == self.nsteps
        shared_actions_n = []
        for i in range(self.world.n):
            if i < self.world.n_adv:
                if self.adv == 'cooperative':
                    shared_actions_n.append(actions[:,:self.world.n_adv])
                if self.adv == 'centralized':
                    shared_actions_n.append(actions[:,:self.world.n_adv])
                if self.adv == 'independent':
                    shared_actions_n.append(actions[:,i:i+1])
            else:
                if self.agt == 'cooperative':
                    shared_actions_n.append(actions[:,self.world.n_adv:])
                if self.agt == 'centralized':
                    shared_actions_n.append(actions[:,self.world.n_adv:])
                if self.agt == 'independent':
                    shared_actions_n.append(actions[:,i:i+1])

        return shared_actions_n

    def save(self):
        for pi in self.policies:
            save_path = pi.manager.save()
            print("Save checkpoint to {}".format(save_path))

    def train(self, actions, obs, returns, dones, values, atarg, neglogpacs):
        A = self.world.comm_matrix
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

        # Policy Update
        if self.adv == 'cooperative':
            argvs = []
            for i in range(self.world.n_adv):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                atarg[i] = (atarg[i] - np.mean(atarg[:self.world.n_adv])) / np.std(atarg[:self.world.n_adv])
                self.policies[i].reinitial_estimates()
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
                argvs.append((obs[i], actions[i], atarg[i], returns[i], values[i]))

            for itr in range(self.admm_iter[0]):
                # edge = adv_edges[np.random.choice(range(len(adv_edges)))]
                edge = adv_edges[itr % len(adv_edges)]
                q = np.where(edge != 0)[0]
                k, j = q[0], q[-1]
                # Update Agent k and j
                self.policies[k].update(*argvs[k])
                self.policies[j].update(*argvs[j])
                ratio_k, multipliers_k = self.policies[k].info_to_exchange(obs[k], actions[k], j)
                ratio_j, multipliers_j = self.policies[j].info_to_exchange(obs[j], actions[j], k)
                self.policies[k].exchange(obs[k], actions[k], edge[k], ratio_j, multipliers_j, j)
                self.policies[j].exchange(obs[j], actions[j], edge[j], ratio_k, multipliers_k, k)
        elif self.adv == 'centralized':
            if self.ob_normalization:
                self.policies[0].pi.ob_rms.update(obs[0])
                self.policies[0].oldpi.ob_rms.update(obs[0])
            atarg[0] = (atarg[0] - np.mean(atarg[0])) / np.std(atarg[0])
            self.policies[0].assign_old_eq_new()
            self.policies[0].vfupdate(obs[0], returns[0], values[0])
            self.policies[0].trpo_update(obs[0], actions[0], atarg[0], returns[0], values[0])
        else:
            for i in range(self.world.n_adv):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                atarg[i] = (atarg[i] - np.mean(atarg[i])) / np.std(atarg[i])
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
                self.policies[i].trpo_update(obs[i], actions[i], atarg[i], returns[i], values[i])

        if self.world.n_agt > 0:
            if self.agt == 'cooperative':
                argvs = []
                for i in range(self.world.n_adv, self.world.n):
                    if self.ob_normalization:
                        self.policies[i].pi.ob_rms.update(obs[i])
                        self.policies[i].oldpi.ob_rms.update(obs[i])
                    atarg[i] = (atarg[i] - np.mean(atarg[self.world.n_adv:])) / np.std(atarg[self.world.n_adv:])
                    self.policies[i].reinitial_estimates()
                    self.policies[i].assign_old_eq_new()
                    self.policies[i].vfupdate(obs[i], returns[i], values[i])
                    argvs.append((obs[i], actions[i], atarg[i], returns[i], values[i]))

                for itr in range(self.admm_iter[1]):
                    # edge = agt_edges[np.random.choice(range(len(agt_edges)))]
                    edge = agt_edges[itr % len(agt_edges)]
                    q = np.where(edge != 0)[0]
                    k, j = q[0], q[-1]
                    nk, nj = k-self.world.n_adv, j-self.world.n_adv
                    # Update Agent k and j
                    self.policies[k].update(*argvs[nk])
                    self.policies[j].update(*argvs[nj])
                    ratio_k, multipliers_k = self.policies[k].info_to_exchange(obs[k], actions[k], nj)
                    ratio_j, multipliers_j = self.policies[j].info_to_exchange(obs[j], actions[j], nk)
                    self.policies[k].exchange(obs[k], actions[k], edge[k], ratio_j, multipliers_j, nj)
                    self.policies[j].exchange(obs[j], actions[j], edge[j], ratio_k, multipliers_k, nk)
            elif self.agt == 'centralized':
                if self.ob_normalization:
                    self.policies[self.world.n_adv].pi.ob_rms.update(obs[self.world.n_adv])
                    self.policies[self.world.n_adv].oldpi.ob_rms.update(obs[self.world.n_adv])
                atarg[self.world.n_adv] = (atarg[self.world.n_adv] - np.mean(atarg[self.world.n_adv])) / np.std(atarg[self.world.n_adv])
                self.policies[self.world.n_adv].assign_old_eq_new()
                self.policies[self.world.n_adv].vfupdate(obs[self.world.n_adv], returns[self.world.n_adv], values[self.world.n_adv])
                self.policies[self.world.n_adv].trpo_update(obs[self.world.n_adv], actions[self.world.n_adv], atarg[self.world.n_adv], returns[self.world.n_adv], values[self.world.n_adv])
            else:
                for i in range(self.world.n_adv, self.world.n):
                    if self.ob_normalization:
                        self.policies[i].pi.ob_rms.update(obs[i])
                        self.policies[i].oldpi.ob_rms.update(obs[i])
                    atarg[i] = (atarg[i] - np.mean(atarg[i])) / np.std(atarg[i])
                    self.policies[i].assign_old_eq_new()
                    self.policies[i].vfupdate(obs[i], returns[i], values[i])
                    self.policies[i].trpo_update(obs[i], actions[i], atarg[i], returns[i], values[i])

