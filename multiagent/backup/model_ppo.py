import tensorflow as tf
import numpy as np
import time

def constfn(val):
    def f(frac):
        return val * frac
    return f

class Model(object):
    def __init__(self, env, world, policies, ncommtime=20, nminibatches=4, noptepochs=4):
        self.env = env
        self.world = world
        self.policies = policies
        self.nsteps = policies[0].nsteps
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.ncommtime = ncommtime

    def step(self, obs_n):
        action_n, value_n, neglogp_n = [], [], []
        for i in range(self.env.n):
            action, value, _, neglogp = self.policies[i].step(obs_n[i])
            action_n.append(np.diag(action._numpy()))
            value_n.append(value)
            neglogp_n.append(neglogp)

        return action_n, value_n, neglogp_n

    def value(self, obs_n):
        return [self.policies[i].value(obs_n[i]) for i in range(self.env.n)]

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

    def train(self, lr, cliprange, obs, returns, dones, actions, values, advs, neglogpacs):
        # Normalize the advantages
        for i in range(self.env.n):
            advs[i] = (advs[i] - np.mean(advs)) / (np.std(advs) + 1e-8)

        lr = constfn(lr)
        cliprange = constfn(cliprange)
        lrnow = tf.constant(lr(1.0))
        cliprangenow = tf.constant(cliprange(1.0))

        # Calculate the batch_size
        nbatch = self.env.nenvs * self.nsteps if hasattr(self.env, 'nenvs') else self.nsteps
        nbatch_train = nbatch // self.nminibatches
        assert nbatch % self.nminibatches == 0

        # Prepare data
        argvs = list(zip(obs, returns, actions, values, advs, neglogpacs))
        exgargvs = list(zip(obs, actions, neglogpacs))

        # # Update value network
        # for i in range(self.env.n):
        #     # Here what we're going to do is for each minibatch calculate the loss and append it.
        #     mblossvals = []
        #     # Index of each element of batch_size
        #     # Create the indices array
        #     inds = np.arange(nbatch)
        #     for _ in range(self.noptepochs):
        #         # Randomize the indexes
        #         np.random.shuffle(inds)
        #         # 0 to batch_size with batch_train_size step
        #         for start in range(0, nbatch, nbatch_train):
        #             end = start + nbatch_train
        #             mbinds = inds[start:end]
        #             slices = (tf.constant(arr[mbinds]) for arr in argvs[i])
        #             mblossvals.append(self.policies[i].vf_update(lrnow, cliprangenow, *slices))
        #     vf_lossvals = np.mean(mblossvals, axis=0)
        #     print('Agent {}: '.format(i), vf_lossvals)

        # reinitialize esitmates
        for i in range(self.env.n):
            self.policies[i].reinitial_estimates()
            self.policies[i].store_oldpi_var()

        # some constants
        A = self.world.comm_matrix
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]


        argvs_k = [tf.constant(arr) for arr in argvs[0]]
        argvs_j = [tf.constant(arr) for arr in argvs[1]]
        _, loss_0, pg_loss_0, sync_loss_0, *_ = self.policies[0].get_pi_grad(
            cliprangenow, 1, self.policies[0].estimates[1], self.policies[0].multipliers[1], *argvs_k)
        _, loss_1, pg_loss_1, sync_loss_1, *_ = self.policies[1].get_pi_grad(
            cliprangenow, 0, self.policies[1].estimates[0], self.policies[1].multipliers[0], *argvs_j)


        for _ in range(self.ncommtime):
            edge = adv_edges[np.random.choice(range(len(adv_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]

            # Prepare training data for agents k and j
            argvs_k = [tf.constant(arr) for arr in argvs[k]]
            argvs_j = [tf.constant(arr) for arr in argvs[j]]

            self.policies[k].assign_new_eq_old()
            self.policies[j].assign_new_eq_old()

            # Exchange info and update estimators
            ratio_k, multipliers_k = self.policies[k].info_to_exchange(cliprangenow, *exgargvs[k], j)
            ratio_j, multipliers_j = self.policies[j].info_to_exchange(cliprangenow, *exgargvs[j], k)
            self.policies[k].exchange(cliprangenow, *exgargvs[k], ratio_j, multipliers_j, j)
            self.policies[j].exchange(cliprangenow, *exgargvs[j], ratio_k, multipliers_k, k)

            # Record initial losses
            _, lossbf_k, pg_lossbf_k, *_ = self.policies[k].get_pi_grad(
                cliprangenow, j, self.policies[k].estimates[j], self.policies[k].multipliers[j], *argvs_k)
            _, lossbf_j, pg_lossbf_j, *_ = self.policies[j].get_pi_grad(
                cliprangenow, k, self.policies[j].estimates[k], self.policies[j].multipliers[k], *argvs_j)

            # Update Agent k
            imp_old = np.array([0.0])
            for itr in range(1, 51):
                frac = 1.0 - (itr - 1.0) / 50
                lrnow = tf.constant(lr(frac))
                loss_k, pg_loss_k, sync_loss_k, *_ = self.policies[k].pi_update(lrnow, cliprangenow, j, *argvs_k)
                print('Agent {}: '.format(k), (lossbf_k-loss_k).numpy(), (pg_lossbf_k-pg_loss_k).numpy(), sync_loss_k.numpy())
                # if sync_loss_k.numpy() < 1e-3:
                #     break
                imp_new = (lossbf_k-loss_k).numpy()
                print(imp_new-imp_old)
                if imp_new-imp_old < -1e-5:
                    continue
                elif imp_new!=0 and imp_new-imp_old < 1e-5:
                    break
                else:
                    imp_old = imp_new.copy()

            # Update Agent j
            imp_old = np.array([0.0])
            for itr in range(1, 11):
                frac = 1.0 - (itr - 1.0) / 10
                lrnow = tf.constant(lr(frac))
                loss_j, pg_loss_j, sync_loss_j, *_ = self.policies[j].pi_update(lrnow, cliprangenow, k, *argvs_j)
                print('Agent {}: '.format(j), (lossbf_j-loss_j).numpy(), (pg_lossbf_j-pg_loss_j).numpy(), sync_loss_j.numpy())
                # if sync_loss_j.numpy() < 1e-3:
                #     break
                imp_new = (lossbf_j-loss_j).numpy()
                print(imp_new-imp_old)
                if imp_new-imp_old < -1e-5:
                    continue
                elif imp_new!=0 and imp_new-imp_old < 1e-5:
                    break
                else:
                    imp_old = imp_new.copy()

            print('----------------------------')
            print(loss_0.numpy(), pg_loss_0.numpy(), sync_loss_0.numpy())
            print(loss_1.numpy(), pg_loss_1.numpy(), sync_loss_1.numpy())
            print('----------------------------')
            print(loss_k.numpy(), pg_loss_k.numpy(), sync_loss_k.numpy())
            print(loss_j.numpy(), pg_loss_j.numpy(), sync_loss_j.numpy())
            # print('----------------------------')
            # print('Agent 0 estimates: \n', self.policies[0].estimates[1])
            # print('Agent 1 estimates: \n', self.policies[1].estimates[0])
            print('----------------------------')
            print('Agent 0 ratios: \n', ratio_k.numpy())
            print('Agent 1 ratios: \n', ratio_j.numpy())
            input()
























        # for _ in range(self.ncommtime):
        #     edge = adv_edges[np.random.choice(range(len(adv_edges)))]
        #     q = np.where(edge != 0)[0]
        #     k, j = q[0], q[-1]
        #     # self.policies[k].assign_new_eq_old()
        #     # self.policies[j].assign_new_eq_old()
        #     argvs_k = [tf.constant(arr) for arr in argvs[k]]
        #     argvs_j = [tf.constant(arr) for arr in argvs[j]]
        #     _, lossbf_k, pg_lossbf_k, *_ = self.policies[k].get_pi_grad(cliprangenow, j, *argvs_k)
        #     _, lossbf_j, pg_lossbf_j, *_ = self.policies[j].get_pi_grad(cliprangenow, k, *argvs_j)
        #     # Update Agent k
        #     imp_old = np.array([0.0, 0.0])
        #     for itr in range(1, 101):
        #         frac = 1.0 - (itr - 1.0) / 100
        #         lrnow = tf.constant(lr(frac))
        #         loss_k, pg_loss_k, sync_loss_k, *_ = self.policies[k].pi_update(lrnow, cliprangenow, j, *argvs_k)
        #         print('Agent {}: '.format(k), (lossbf_k-loss_k).numpy(), (pg_lossbf_k-pg_loss_k).numpy(), sync_loss_k.numpy())
        #         loss_j, pg_loss_j, sync_loss_j, *_ = self.policies[j].pi_update(lrnow, cliprangenow, k, *argvs_j)
        #         print('Agent {}: '.format(j), (lossbf_j-loss_j).numpy(), (pg_lossbf_j-pg_loss_j).numpy(), sync_loss_j.numpy())
        #         # print(lrnow, cliprangenow)
        #         # print('Agent {} estimates: '.format(k), self.policies[k].estimates)
        #         # print('Agent {} estimates: '.format(j), self.policies[j].estimates)
        #         imp_new = np.hstack([(lossbf_k-loss_k).numpy(), (lossbf_j-loss_j).numpy()])
        #         print(imp_new-imp_old)
        #         if np.any(imp_new-imp_old < -1e-5):
        #             continue
        #         elif np.all(imp_new!=0) and np.all(imp_new-imp_old < 1e-5):
        #             ratio_k, multipliers_k = self.policies[k].info_to_exchange(*exgargvs[k], j)
        #             ratio_j, multipliers_j = self.policies[j].info_to_exchange(*exgargvs[j], k)
        #             self.policies[k].exchange(*exgargvs[k], ratio_j, multipliers_j, j)
        #             self.policies[j].exchange(*exgargvs[j], ratio_k, multipliers_k, k)
        #             break
        #         else:
        #             imp_old = imp_new.copy()
        #         # input()
        #         # time.sleep(.5)
        #     else:
        #         ratio_k, multipliers_k = self.policies[k].info_to_exchange(*exgargvs[k], j)
        #         ratio_j, multipliers_j = self.policies[j].info_to_exchange(*exgargvs[j], k)
        #         self.policies[k].exchange(*exgargvs[k], ratio_j, multipliers_j, j)
        #         self.policies[j].exchange(*exgargvs[j], ratio_k, multipliers_k, k)

        #     print('----------------------------')
        #     print(loss_0.numpy(), pg_loss_0.numpy(), sync_loss_0.numpy())
        #     print(loss_1.numpy(), pg_loss_1.numpy(), sync_loss_1.numpy())
        #     print('----------------------------')
        #     print(loss_k.numpy(), pg_loss_k.numpy(), sync_loss_k.numpy())
        #     print(loss_j.numpy(), pg_loss_j.numpy(), sync_loss_j.numpy())
        #     print('----------------------------')
        #     print('Agent 0 estimates: \n', self.policies[0].estimates[1])
        #     print('Agent 1 estimates: \n', self.policies[1].estimates[0])
        #     print('----------------------------')
        #     print('Agent 0 logits: \n', ratio_k.numpy())
        #     print('Agent 1 logits: \n', ratio_j.numpy())
        #     input()
