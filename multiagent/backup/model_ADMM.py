import time
import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import zipsame, dataset, colorize
from contextlib import contextmanager

class Model(object):
    def __init__(self, env, world, policies, nsteps, load_path=None, admm_iter=10,
                 ent_coef=0.0, vf_iters=3, alpha=1.0, rho=100.0, beta=1.0, eta=1.0):
        self.sess = sess = U.get_session()
        self.env = env
        self.world = world
        self.vf_iters = vf_iters
        self.admm_iter = admm_iter
        if hasattr(env, 'num_envs'):
            self.n_batches = n_batches = nsteps * env.num_envs
        else:
            self.n_batches = n_batches = nsteps

        # GLOBAL PLACEHOLDERS
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        self.LR = LR = tf.placeholder(tf.float32, [])
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        self.OB_n, self.AC_n, self.ADV_n, self.R_n = [], [], [], []
        self.COMM_n, self.OLDVPRED_n, self.NB_n = [], [], []
        self.pi_n, self.oldpi_n, self.pg_train_n, self.vf_train_n = [], [], [], []
        self.exchange_n, self.to_exchange_n, self.losses_n = [], [], []
        for i in range(world.n):
            name_scope = world.agents[i].name.replace(' ', '')
            with tf.variable_scope(name_scope):
                # OBSERVATION PLACEHOLDER
                ob_dtype = env.observation_space[i].dtype
                ob_shape = env.observation_space[i].shape
                OB = tf.placeholder(dtype=ob_dtype, shape=(None,)+ob_shape)
                # Policy
                with tf.variable_scope("pi"):
                    pi = policies[i](n_batches, observ_placeholder=OB)
                with tf.variable_scope("oldpi"):
                    oldpi = policies[i](n_batches, observ_placeholder=OB)

                # CREATE OTHER PLACEHOLDERS
                AC = pi.pdtype.sample_placeholder([None])
                ADV = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                R = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                NB = tf.placeholder(dtype=tf.int32, shape=None)
                COMM = tf.placeholder(dtype=tf.float32, shape=None)
                OLDVPRED = tf.placeholder(dtype=tf.float32, shape=[None, 1])

                # Calculate ratio (pi current policy / pi old policy)
                ratio = tf.expand_dims(tf.exp(pi.pd.logp(AC) - oldpi.pd.logp(AC)), axis=1)
                pg_loss1 = - ADV * ratio
                pg_loss2 = - ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
                entropy = tf.reduce_mean(pi.pd.entropy())
                pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2)) - entropy * ent_coef
                grad, logit = self.trainer.compute_gradients(pg_loss, pi.logit)[0]
                target_logit = logit - LR * grad
                pg_train_op = pi.fit(target_logit, COMM, NB, alpha, rho, beta, eta)

                # Define the value loss
                vpredclipped = OLDVPRED + tf.clip_by_value(pi.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
                # vpredclipped = tf.clip_by_value(pi.vf, OLDVPRED*(1-CLIPRANGE), OLDVPRED*(1+CLIPRANGE))
                vf_loss1 = tf.square(pi.vf - R)
                vf_loss2 = tf.square(vpredclipped - R)
                vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
                vf_train_op = self.trainer.minimize(vf_loss)

                # vfupdate_op = pi.fit_vf(R, gamma, beta)
                vf_train = U.function([OB, R, OLDVPRED, CLIPRANGE, LR], [vf_loss], updates=[vf_train_op])
                pg_train = U.function([OB, AC, ADV, CLIPRANGE, COMM, NB, LR], [], updates=pg_train_op)
                exchange = pi.net.exchange(sess, OB, NB, rho)
                to_exchange = U.function([OB, NB], pi.net.info_to_exchange(NB))
                losses = U.function([OB, AC, ADV, CLIPRANGE], [pg_loss])

            self.OB_n.append(OB)
            self.AC_n.append(AC)
            self.ADV_n.append(ADV)
            self.R_n.append(R)
            self.OLDVPRED_n.append(OLDVPRED)
            self.COMM_n.append(COMM)
            self.NB_n.append(NB)
            self.vf_train_n.append(vf_train)
            self.pg_train_n.append(pg_train)
            self.to_exchange_n.append(to_exchange)
            self.exchange_n.append(exchange)
            self.losses_n.append(losses)
            self.pi_n.append(pi)
            self.oldpi_n.append(oldpi)

        # Update old plicy network
        updates = []
        for i in range(world.n):
            name_scope = world.agents[i].name.replace(' ', '')
            old_vars = get_trainable_variables("{}/oldpi".format(name_scope))
            now_vars = get_trainable_variables("{}/pi".format(name_scope))
            updates += [tf.assign(oldv, newv) for (oldv, newv) in zipsame(old_vars, now_vars)]
        self.assign_old_eq_new = U.function([],[], updates=updates)

        @contextmanager
        def timed(msg):
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        self.timed = timed

        # Initialization
        U.initialize()
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        variables = []
        for agent in self.world.agents:
            name_scope = agent.name.replace(' ', '')
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
        U.load_variables(load_path, variables=variables, sess=self.sess)

    def save(self, save_path):
        variables = []
        for agent in self.world.agents:
            name_scope = agent.name.replace(' ', '')
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
        U.save_variables(save_path, variables=variables, sess=self.sess)

    def step(self, obs_n, nenv=1):
        if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
        actions_n, values_n = zip(*[self.pi_n[i].step(obs_n[i]) for i in range(self.world.n)])
        actions_n = np.hstack(actions_n)
        values_n = np.hstack(values_n)
        if nenv == 1: actions_n = actions_n.squeeze()

        return actions_n, values_n

    def value(self, obs_n, nenv):
        if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
        return np.hstack([self.pi_n[i].value(obs_n[i]) for i in range(self.world.n)])

    def share_actions(self, actions_n):
        shared_actions_n = list(actions_n)
        for i in range(self.world.n_adv):
            shared_actions_n[i] = np.hstack(actions_n[:self.world.n_adv]).astype(actions_n[i].dtype).squeeze()
        for i in range(self.world.n_adv, self.world.n):
            shared_actions_n[i] = np.hstack(actions_n[self.world.n_adv:]).astype(actions_n[i].dtype).squeeze()

        return shared_actions_n

    def train_adv(self, iters, cliprange, lr, obs_n, actions_n, mb_rewards, values_n, advs_n, returns_n, dones_n):
        # some constants
        eps = 1e-8
        A = self.world.A
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

        # Set old parameter values to new parameter values
        self.assign_old_eq_new()

        # Prepare data
        advs_n = [(advs_n[i] - advs_n[i].mean()) / (advs_n[i].std() + eps) for i in range(self.world.n_adv)]
        args_n = [(obs_n[i], actions_n[i], advs_n[i], cliprange) for i in range(self.world.n_adv)]

        # Train adversaries
        loss_bf = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv)]))
        for itr in range(self.admm_iter):
            edge = adv_edges[np.random.choice(range(len(adv_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]
            # print('The {}th iteration of adversaries!'.format(itr))
            self.pg_train_n[k](*args_n[k], edge[k], j, lr)
            self.pg_train_n[j](*args_n[j], edge[j], k, lr)
            if len(q) > 1:
                a_k, p_k = self.to_exchange_n[k](obs_n[k], j)
                a_j, p_j = self.to_exchange_n[j](obs_n[j], k)
                self.exchange_n[k](j, a_j, p_j, edge[j], obs_n[k], edge[k])
                self.exchange_n[j](k, a_k, p_k, edge[k], obs_n[j], edge[j])
            # Train value function
            for _ in range(self.vf_iters):
                argvs_k = (obs_n[k], returns_n[k], values_n[k])
                for (mbob, mbret, mbvl) in dataset.iterbatches(argvs_k,
                include_final_partial_batch=False, batch_size=64):
                    self.vf_train_n[k](mbob, mbret, mbvl, cliprange, lr)
                argvs_j = (obs_n[j], returns_n[j], values_n[j])
                for (mbob, mbret, mbvl) in dataset.iterbatches(argvs_j,
                include_final_partial_batch=False, batch_size=64):
                    self.vf_train_n[j](mbob, mbret, mbvl, cliprange, lr)

            loss_itr = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv)]))
            imp = np.array(loss_bf).ravel() - np.array(loss_itr).ravel()
            print('  Inner iteration {}: {}'.format(itr, imp))

        # # # Train adversaries
        # # with self.timed("pi"):
        # #     for itr in range(iters):
        # #         self.ppo_train_n[0](*args[0])
        # #         loss_itr, grad_itr = list(zip(*[self.lossandgrad_n[i](*args[i]) for i in range(n_adv)]))
        # #         print(losses[0]-loss_itr[0])
        # #         name_scope = self.env.agents[i].name.replace(' ', '')
        # #         w0 = self.sess.run(get_trainable_variables("{}/pi/pi/w0".format(name_scope))[0])
        # #         w1 = self.sess.run(get_trainable_variables("{}/pi/pi/w1".format(name_scope))[0])
        # #         w2 = self.sess.run(get_trainable_variables("{}/pi/pi/w2".format(name_scope))[0])
        # #         print('max_w0: ', np.abs(w0).max())
        # #         print('max_w1: ', np.abs(w1).max())
        # #         print('max_w2: ', np.abs(w2).max())
        # #         input()

        # # Train adversaries
        # with self.timed("pi"):
        #     adv_imp, expire_counter = 0, 0
        #     for itr in range(iters):
        #     # wgile True:
        #         # print('The {}th iteration of adversaries!'.format(itr))
        #         for edge in adv_edges:
        #             q = np.where(edge != 0)[0]
        #             k, j = q[0], q[-1]
        #             self.update_n[k](*args[k], grad_n[k], oldmu_n[k], edge[k], j)
        #             self.update_n[j](*args[j], grad_n[j], oldmu_n[j], edge[j], k)
        #             # Notice: self.to_exchange_n must run after self.update_n
        #             if len(q)>1:
        #                 a_k, p_k = self.to_exchange_n[k](j)
        #                 a_j, p_j = self.to_exchange_n[j](k)
        #                 self.exchange_n[k](j, a_j, p_j, edge[j], edge[k])
        #                 self.exchange_n[j](k, a_k, p_k, edge[k], edge[j])

        #         loss_itr, grad_itr, oldmu_itr = list(zip(*[self.lossandgrad_n[i](*args[i]) for i in range(n_adv)]))
        #         name_scope = self.env.agents[i].name.replace(' ', '')
        #         s = self.sess.run(self.pi_n[0].net.s, feed_dict={self.OB_n[0]: obs_n[0]})
        #         w0 = self.sess.run(get_trainable_variables("{}/pi/pi/w0".format(name_scope))[0])
        #         w1 = self.sess.run(get_trainable_variables("{}/pi/pi/w1".format(name_scope))[0])
        #         w2 = self.sess.run(get_trainable_variables("{}/pi/pi/w2".format(name_scope))[0])
        #         x0 = self.sess.run(get_trainable_variables("{}/pi/pi/x0".format(name_scope))[0])
        #         x1 = self.sess.run(get_trainable_variables("{}/pi/pi/x1".format(name_scope))[0])
        #         o0 = self.sess.run(get_trainable_variables("{}/pi/pi/o0".format(name_scope))[0])
        #         o1 = self.sess.run(get_trainable_variables("{}/pi/pi/o1".format(name_scope))[0])
        #         a = self.sess.run(get_trainable_variables("{}/pi/pi/a".format(name_scope))[0])
        #         mu = self.sess.run(self.pi_n[0].net.policy(), feed_dict={self.OB_n[0]: obs_n[0]})
        #         # print('max_w0: ', np.abs(w0).max())
        #         # print('max_w1: ', np.abs(w1).max())
        #         # print('max_w2: ', np.abs(w2).max())
        #         # print('max_o1: ', np.abs(o1).max())
        #         # print('w0*s - x0: ', np.abs(w0.dot(s)-x0).max())
        #         # print('w1*o0 - x1: ', np.abs(w1.dot(o0)-x1).max())
        #         # print('w2*o1 - a: ', np.abs(w2.dot(o1)-a).max())
        #         err = np.sum(np.square(mu - (oldmu_n[0] - 0.01 * grad_n[0])))
        #         imp = np.array(loss_n) - np.array(loss_itr)
        #         print(imp, err)
        #         # print('oldmu - grad: ', )
        #         # input()
        #         # time.sleep(.5)

        # # input()

        # # losses_update, grads_update = [], []
        # # for i in range(n_adv):
        # #     loss, grad = self.lossandgrad_n[i](*args[i])
        # #     losses_update.append(loss)
        # #     grads_update.append(grad)

        # if np.any(imp < 0):
        #     print('Rollback!')
        #     # self.rollback()


    def train_agt(self, iters, cliprange, lr, obs_n, actions_n, mb_rewards, values_n, advs_n, returns_n, dones_n):
        # some constants
        eps = 1e-8
        A = self.world.A
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

        # Set old parameter values to new parameter values
        self.assign_old_eq_new()

        # Prepare data
        advs_n = [(advs_n[i] - advs_n[i].mean()) / (advs_n[i].std() + eps) for i in range(self.world.n)]
        args_n = [(obs_n[i], actions_n[i], advs_n[i], cliprange) for i in range(self.world.n)]

        # Train good agents
        loss_bf, logit_bf = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
        for itr in range(self.admm_iter):
            edge = adv_edges[np.random.choice(range(len(adv_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]
            nk, nj = k - self.world.n_adv, k - self.world.n_adv
            # print('The {}th iteration of adversaries!'.format(itr))
            self.pg_train_n[k](*args_n[k], edge[k], nj, lr)
            self.pg_train_n[j](*args_n[j], edge[j], nk, lr)
            if len(q) > 1:
                a_k, p_k = self.to_exchange_n[k](obs_n[k], nj)
                a_j, p_j = self.to_exchange_n[j](obs_n[j], nk)
                self.exchange_n[k](j, a_j, p_j, edge[j], obs_n[k], edge[k])
                self.exchange_n[j](k, a_k, p_k, edge[k], obs_n[j], edge[j])
            # Train value function
            for _ in range(self.vf_iters):
                argvs_k = (obs_n[k], returns_n[k], values_n[k])
                for (mbob, mbret, mbvl) in dataset.iterbatches(argvs_k,
                include_final_partial_batch=False, batch_size=64):
                    self.vf_train_n[k](mbob, mbret, mbvl, cliprange, lr)
                argvs_j = (obs_n[j], returns_n[j], values_n[j])
                for (mbob, mbret, mbvl) in dataset.iterbatches(argvs_j,
                include_final_partial_batch=False, batch_size=64):
                    self.vf_train_n[j](mbob, mbret, mbvl, cliprange, lr)

            loss_itr = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
            imp = np.array(loss_bf).ravel() - np.array(loss_itr).ravel()
            print('  Inner iteration {}: {}'.format(itr, imp))

            # # Check improvement
            # loss, grad = list(zip(*[self.lossandgrad_n[i](*args[i]) for i in range(n_adv, self.env.n)]))
            # imp = agt_loss_0 - sum(loss)
            # if imp == agt_imp:
            #     expire_counter += 1
            # else:
            #     agt_imp = imp
            #     expire_counter = 0
            # print(imp, agt_imp, sum(loss), expire_counter)
            # Stop training if not improving
            # if expire_counter >= 5: break


            # # print(m, q)
            # if np.all(q < n_adv):
            #     i, j = q[0], q[1]
            #     name_scope = self.env.agents[i].name.replace(' ', '')
            #     # w0 = model.sess.run(get_trainable_variables("{}/pi/pi/w0".format(name_scope))[0])
            #     # x0 = self.sess.run(get_trainable_variables("{}/pi/pi/x0".format(name_scope))[0])
            #     a = self.sess.run(get_trainable_variables("{}/pi/pi/a".format(name_scope))[0])
            #     # z = self.sess.run(get_trainable_variables("{}/pi/pi/z".format(name_scope))[0])
            #     # p = self.sess.run(get_trainable_variables("{}/pi/pi/p".format(name_scope))[0])
            #     # lam = self.sess.run(get_trainable_variables("{}/pi/pi/lam".format(name_scope))[0])
            #     # w1 = model.sess.run(get_trainable_variables("{}/pi/pi/w1".format(name_scope))[0])
            #     # w2 = model.sess.run(get_trainable_variables("{}/pi/pi/w2".format(name_scope))[0])
            #     # w3 = model.sess.run(get_trainable_variables("{}/pi/pi/w3".format(name_scope))[0])
            #     # w4 = model.sess.run(get_trainable_variables("{}/pi/pi/w4".format(name_scope))[0])
            #     # print('max_w0: ', np.abs(w0).max())
            #     # print('max_w1: ', np.abs(w1).max())
            #     # print('max_w2: ', np.abs(w2).max())
            #     # print('max_w3: ', np.abs(w3).max())
            #     # print('max_w4: ', np.abs(w4).max())
            #     # print('max_x0, min_x0: ', np.abs(x0).max(), np.abs(x0).min())
            #     # print('max_a, min_a: ', np.abs(a).max(), np.abs(a).min())
            #     # print('max_z, min_z: ', np.abs(z).max(), np.abs(z).min())
            #     # print('max_p, min_p: ', np.abs(p).max(), np.abs(p).min())
            #     # print('max_lam, min_lam: ', np.abs(lam).max(), np.abs(lam).min())
            #     # print('w1: ', self.sess.run(tf.linalg.det(tf.matmul(w1, w1, transpose_a=True))))
            #     # print('w2: ', self.sess.run(tf.linalg.det(tf.matmul(w2, w2, transpose_a=True))))
            #     # print('w3: ', self.sess.run(tf.linalg.det(tf.matmul(w3, w3, transpose_a=True))))
            #     # print('w4: ', self.sess.run(tf.linalg.det(tf.matmul(w4, w4, transpose_a=True))))
            #     # input()
            #     self.update_n[i](*args[i], j, A[m,i], mu[i](), old_mu[i])
            #     self.update_n[j](*args[j], i, A[m,j], mu[j](), old_mu[j])
            #     # Notice: self.to_exchange_n must run after self.update_n
            #     a_i, p_i = self.to_exchange_n[i](j)
            #     a_j, p_j = self.to_exchange_n[j](i)
            #     self.exchange_n[i](j, a_j, p_j, A[m,i], A[m,j])
            #     self.exchange_n[j](i, a_i, p_i, A[m,j], A[m,i])
            #     # update value network
            #     for _ in range(self.vf_iters):
            #         vfloss = self.vfupdate_n[i](obs_n[i], returns_n[i])
            #         vfloss = self.vfupdate_n[j](obs_n[j], returns_n[j])
            # elif np.all(q >= n_adv):
            #     if len(q) == 1:
            #         q = q[0]
            #         self.update_n[q](*args[q], 0, A[m,q], mu[q](), old_mu[q])
            #         # update value network
            #         for _ in range(self.vf_iters):
            #             vfloss = self.vfupdate_n[q](obs_n[q], returns_n[q])
            #     else:
            #         i, j = q[0], q[1]
            #         ni, nj = i-n_adv, j-n_adv
            #         # prepare data
            #         self.update_n[i](*args[i], nj, A[m,i], mu[i](), old_mu[i])
            #         self.update_n[j](*args[j], ni, A[m,j], mu[j](), old_mu[j])
            #         # Notice: self.to_exchange_n must run after self.update_n
            #         a_i, p_i = self.to_exchange_n[i](nj)
            #         a_j, p_j = self.to_exchange_n[j](ni)
            #         self.exchange_n[i](nj, a_j, p_j, A[m,i], A[m,j])
            #         self.exchange_n[j](ni, a_i, p_i, A[m,j], A[m,i])
            #         # update value network
            #         for _ in range(self.vf_iters):
            #             self.vfupdate_n[i](obs_n[i], returns_n[i])
            #             self.vfupdate_n[j](obs_n[j], returns_n[j])
            # else:
            #     raise 'Communication bewteen good agents and adversary agents are not allowed!'

            # adv_loss = sum([self.lossandgrad_n[i](*args[i], mu[i](), old_mu[i])[0] for i in range(n_adv)])
            # agt_loss = sum([self.lossandgrad_n[i](*args[i], mu[i](), old_mu[i])[0] for i in range(n_adv, self.env.n)])
            # print([adv_loss-adv_loss_0, agt_loss-agt_loss_0, adv_loss, agt_loss])


def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
