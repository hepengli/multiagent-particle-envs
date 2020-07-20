import time
import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import zipsame, dataset, colorize
from contextlib import contextmanager
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from copy import deepcopy

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class Model(object):
    def __init__(self, env, world, policies, nsteps, load_path, rho, max_kl,
                 ent_coef, vf_coef, max_grad_norm, sync):
        self.sess = sess = U.get_session()
        self.env = env
        self.world = world
        self.sync = sync
        self.max_kl = max_kl
        if hasattr(env, 'num_envs'):
            self.n_batches = n_batches = nsteps * env.num_envs
        else:
            self.n_batches = n_batches = nsteps

        if MPI is not None:
            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.nworkers = 1
            self.rank = 0

        cpus_per_worker = 1
        U.get_session(config=tf.ConfigProto(
                allow_soft_placement=True,
                inter_op_parallelism_threads=cpus_per_worker,
                intra_op_parallelism_threads=cpus_per_worker
        ))

        # GLOBAL PLACEHOLDERS
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        self.pi_n, self.oldpi_n, self.vfadam_n, self.exchange_n, self.to_exchange_n = [], [], [], [], []
        self.compute_jtvp_n, self.compute_fvp_n, self.compute_losses_n, self.compute_vfloss_n = [], [], [], []
        self.set_from_flat_n, self.get_flat_n = [], []
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
                ADV = tf.placeholder(dtype=tf.float32, shape=[None])
                R = tf.placeholder(dtype=tf.float32, shape=[None])
                OLDVPRED = tf.placeholder(dtype=tf.float32, shape=[None])
                NB = tf.placeholder(dtype=tf.int32, shape=None)
                A = tf.placeholder(dtype=tf.float32, shape=None)

                ratio = tf.exp(pi.pd.logp(AC) - oldpi.pd.logp(AC)) # Be careful about the dimensionality!!!!!!!!!!!!!!!!
                surrgain = tf.reduce_mean(ADV * ratio)
                kloldnew = oldpi.pd.kl(pi.pd)
                meankl = tf.reduce_mean(kloldnew)
                sync_err = A * tf.reshape(ratio, (self.n_batches,)) - tf.reshape(tf.gather(pi.net.z, NB), (self.n_batches,))
                sync_loss = tf.reduce_sum(tf.reshape(tf.gather(pi.net.z, NB), (self.n_batches,)) * sync_err) + \
                            0.5 * rho * tf.reduce_sum(tf.square(sync_err))
                lagrange_loss = - surrgain + sync_loss
                losses = [lagrange_loss, surrgain, meankl]
                dist = meankl

                var_list = pi.net.w
                klgrads = tf.gradients(dist, var_list)
                flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")

                shapes = [var.get_shape().as_list() for var in var_list]
                start = 0
                tangents = []
                for shape in shapes:
                    sz = U.intprod(shape)
                    tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
                    start += sz

                jjvp = [tf.zeros(shape, dtype=tf.float32) for shape in shapes]
                jtvp = [tf.zeros(shape, dtype=tf.float32) for shape in shapes]
                right_b = - ADV + A * tf.gather(pi.net.p, NB) - rho * A * tf.gather(pi.net.z, NB)
                for i in range(self.n_batches):
                    ratio_i_grad = tf.gradients(ratio[i], var_list)
                    jvp_i = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(ratio_i_grad, tangents)])
                    jjvp = [tf.add_n([jj, gg*jvp_i]) for (jj, gg) in zipsame(jjvp, ratio_i_grad)]
                    jtvp = [tf.add_n([jt, gt*right_b[i]]) for (jt, gt) in zipsame(jtvp, ratio_i_grad)]
                    print(i)

                jjvp = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in jjvp])
                jtvp = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in jtvp])
                gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
                fvp = tf.add_n([U.flatgrad(gvp, var_list), rho * jjvp])

                # Define the value loss
                vpredclipped = OLDVPRED + tf.clip_by_value(pi.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
                # vpredclipped = tf.clip_by_value(pi.vf, OLDVPRED*(1-CLIPRANGE), OLDVPRED*(1+CLIPRANGE))
                vferr = tf.square(pi.vf - R)
                vferr2 = tf.square(vpredclipped - R)
                vf_loss = .5 * tf.reduce_mean(tf.maximum(vferr, vferr2))
                vfadam = MpiAdam(pi.net.v)

                compute_jtvp = U.function([OB, AC, ADV, A, NB], jtvp)
                compute_fvp = U.function([flat_tangent, OB, AC, ADV], fvp)
                compute_losses = U.function([OB, AC, ADV, A, NB], losses)
                compute_vfloss = U.function([OB, R, OLDVPRED, CLIPRANGE], vf_loss)
                exchange = pi.net.exchange(sess, OB, AC, CLIPRANGE, NB, rho)
                to_exchange = U.function([OB, AC, ADV, NB, CLIPRANGE], [ratio, tf.gather(pi.net.p, NB)])

                get_flat = U.GetFlat(var_list)
                set_from_flat = U.SetFromFlat(var_list)

            self.pi_n.append(pi)
            self.oldpi_n.append(oldpi)
            self.get_flat_n.append(get_flat)
            self.set_from_flat_n.append(set_from_flat)
            self.vfadam_n.append(vfadam)
            self.exchange_n.append(exchange)
            self.to_exchange_n.append(to_exchange)
            self.compute_jtvp_n.append(compute_jtvp)
            self.compute_fvp_n.append(compute_fvp)
            self.compute_losses_n.append(compute_losses)
            self.compute_vfloss_n.append(compute_vfloss)

        # Update old plicy network
        updates = []
        for i in range(len(world.agents)):
            name_scope = world.agents[i].name.replace(' ', '')
            old_vars = get_trainable_variables("{}/oldpi".format(name_scope))
            now_vars = get_trainable_variables("{}/pi".format(name_scope))
            updates += [tf.assign(oldv, nowv) for (oldv, nowv) in zipsame(old_vars, now_vars)]
            updates += [tf.assign(self.pi_n[i].net.z, tf.ones_like(self.pi_n[i].net.z))]
        self.assign_old_eq_new = U.function([],[], updates=updates)

        @contextmanager
        def timed(msg):
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        self.timed = timed

        def allmean(x):
            assert isinstance(x, np.ndarray)
            if MPI is not None:
                out = np.empty_like(x)
                MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
                out /= self.nworkers
            else:
                out = np.copy(x)

            return out
        self.allmean = allmean

        # Initialization
        U.initialize()
        if load_path is not None:
            self.load(load_path)

        # for i in range(len(self.pi_n)):
            th_init = self.get_flat_n[i]()
            self.set_from_flat_n[i](th_init)
            print("Init param sum", th_init.sum(), flush=True)

        for vfadam in self.vfadam_n:
            vfadam.sync()

    def load(self, load_path):
        # variables = []
        # for agent in self.world.agents:
        #     name_scope = agent.name.replace(' ', '')
        #     variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
        # U.load_variables(load_path, variables=variables, sess=self.sess)
        for pi in self.pi_n:
            pi.load(load_path)

    def save(self, save_path):
        # variables = []
        # for agent in self.world.agents:
        #     name_scope = agent.name.replace(' ', '')
        #     variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
        # U.save_variables(save_path, variables=variables, sess=self.sess)
        for pi in self.pi_n:
            pi.save(save_path)

    def print_variables(self):
        for i, pi in enumerate(self.pi_n):
            w = self.sess.run(pi.net.w)
            print('------------------------------')
            print('Agent {}:'.format(i))
            print(w)


    # def step(self, obs_n, nenv=1):
    #     if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
    #     actions_n, values_n = [], []
    #     for i in range(self.world.n):
    #         actions, values = self.pi_n[i].step(obs_n[i])
    #         actions_n.append(actions)
    #         values_n.append(values)
    #     # print(actions_n, values_n)
    #     actions_n = np.hstack(actions_n)
    #     values_n = np.hstack(values_n)
    #     # print(actions_n, values_n)
    #     # input()
    #     if nenv == 1: actions_n = actions_n.squeeze()

    #     return actions_n, values_n

    # def value(self, obs_n, nenv):
    #     if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
    #     return np.hstack([self.pi_n[i].value(obs_n[i]) for i in range(self.world.n)])

    # def share_actions(self, actions_n):
    #     shared_actions_n = list(actions_n)
    #     for i in range(self.world.n_adv):
    #         shared_actions_n[i] = np.hstack(actions_n[:self.world.n_adv]).astype(actions_n[i].dtype).squeeze()
    #     for i in range(self.world.n_adv, self.world.n):
    #         shared_actions_n[i] = np.hstack(actions_n[self.world.n_adv:]).astype(actions_n[i].dtype).squeeze()

    #     return shared_actions_n
    def step(self, obs_n):
        return zip(*[self.pi_n[i].step(obs_n[i]) for i in range(self.env.n)])

    def value(self, obs_n):
        return [self.pi_n[i].value(obs_n[i]) for i in range(self.env.n)]

    def share_actions(self, actions_n):
        shared_actions_n = list(actions_n)
        if hasattr(self.env.agents[0], 'adversary'):
            n_a = self.env.world.num_adversaries
            n_g = self.env.world.num_good_agents
        else:
            n_a = len(self.env.agents)
            n_g= 0

        for i in range(n_a):
            shared_actions_n[i] = np.hstack(actions_n[:n_a]).astype(actions_n[i].dtype).squeeze()
        for i in range(n_g):
            shared_actions_n[n_a+i] = np.hstack(actions_n[n_a:]).astype(actions_n[n_a+i].dtype).squeeze()

        return shared_actions_n

    def train_adv(self, iters, cliprange, lr, obs_n, actions_n, values_n, advs_n, returns_n, dones_n):
        # some constants
        eps = 1e-8
        A = self.world.A
        adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
        agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

        for i, pi in enumerate(self.pi_n):
            if hasattr(pi, "ret_rms"): pi.ret_rms.update(returns_n[i])
            if hasattr(pi, "ob_rms"): pi.ob_rms.update(obs_n[i]) # update running mean/std for policy

        # Set old parameter values to new parameter values
        self.assign_old_eq_new()

        # Prepare data
        advs_n = [(advs_n[i] - advs_n[i].mean()) / (advs_n[i].std() + eps) for i in range(self.world.n_adv)]
        for i in range(self.world.n_adv):
            obs_n[i] = np.squeeze(obs_n[i])
            actions_n[i] = np.squeeze(actions_n[i])
            values_n[i] = np.squeeze(values_n[i])
            advs_n[i] = np.squeeze(advs_n[i])
            returns_n[i] = np.squeeze(returns_n[i])
        # for i in range(self.world.n_adv):
        #     print(obs_n[i].shape, actions_n[i].shape, values_n[i].shape, advs_n[i].shape, returns_n[i].shape)
        #     input()

        # Train adversaries
        for itr in range(iters):
            edge = adv_edges[np.random.choice(range(len(adv_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]
            if k == j: # Agent is alone
                # for _ in range(self.admm_iter):
                #     self.train_n[k](*args_n[k], 0, j, lr)
                pass
            else:
                def fisher_vector_product(p):
                    return self.allmean(self.compute_fvp_n[k](p, obs_n[k], actions_n[k], advs_n[k])) + 0.02 * p
                g_k = self.compute_jtvp_n[k](obs_n[k], actions_n[k], advs_n[k], edge[k], j)
                g_k = self.allmean(g_k)
                if np.allclose(g_k, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with self.timed("cg"):
                        stepdir = cg(fisher_vector_product, g_k, cg_iters=10, verbose=self.rank==0)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / self.max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g_k.dot(fullstep)

                    loss_bf, surr_bf, kl_bf = self.allmean(np.array(self.compute_losses_n[i](obs_n[k], actions_n[k], advs_n[k], edge[k], j)))
                    stepsize = 1.0
                    thbefore = self.get_flat_n[k]()
                    w_bf = self.sess.run(self.pi_n[k].net.w)
                    for _ in range(10):
                        thnew = thbefore - fullstep * stepsize
                        self.set_from_flat_n[k](thnew)
                        meanlosses = loss, surr, kl = self.allmean(np.array(self.compute_losses_n[i](obs_n[k], actions_n[k], advs_n[k], edge[k], j)))
                        improve = loss_bf - loss
                        print(improve)
                        input()
                        logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl > self.max_kl * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        self.set_from_flat_n[k](thbefore)

                def fisher_vector_product(p):
                    return self.allmean(self.compute_fvp_n[j](p, obs_n[j], actions_n[j], advs_n[j])) + 0.02 * p
                g_j = self.compute_jtvp_n[j](obs_n[j], actions_n[j], advs_n[j], edge[j], k)
                g_j = self.allmean(g_j)
                if np.allclose(g_j, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with self.timed("cg"):
                        stepdir = cg(fisher_vector_product, g_j, cg_iters=10, verbose=self.rank==0)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / self.max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g_j.dot(fullstep)

                    loss_bf, surr_bf, kl_bf = self.allmean(np.array(self.compute_losses_n[i](obs_n[j], actions_n[j], advs_n[j], edge[j], k)))
                    stepsize = 1.0
                    thbefore = self.get_flat_n[j]()
                    w_bf = self.sess.run(self.pi_n[j].net.w)
                    for _ in range(10):
                        thnew = thbefore - fullstep * stepsize
                        self.set_from_flat_n[j](thnew)
                        meanlosses = loss, surr, kl = self.allmean(np.array(self.compute_losses_n[i](obs_n[j], actions_n[j], advs_n[j], edge[j], k)))
                        improve = loss_bf - loss
                        print(improve)
                        input()
                        logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl > self.max_kl * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        self.set_from_flat_n[j](thbefore)

                if self.sync:
                    a_k, p_k = self.to_exchange_n[k](obs_n[k], actions_n[k], advs_n[k], cliprange, j)
                    a_j, p_j = self.to_exchange_n[j](obs_n[j], actions_n[j], advs_n[j], cliprange, k)
                    self.exchange_n[k](j, a_k, a_j, p_k, p_j, edge[j], obs_n[k], actions_n[k], cliprange, edge[k])
                    self.exchange_n[j](k, a_j, a_k, p_j, p_k, edge[k], obs_n[j], actions_n[j], cliprange, edge[j])

                    z_k = self.sess.run(self.pi_n[k].net.z[j])
                    z_j = self.sess.run(self.pi_n[j].net.z[k])
                    print('zk + zj: {}'.format(np.sum(np.square(z_k+z_j))))

            print('  Inner iteration {}'.format(itr))

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
        args_n = [(obs_n[i], actions_n[i], advs_n[i], returns_n[i], values_n[i], cliprange) for i in range(self.world.n)]

        # Train good agents
        loss_bf, vfloss_bf, entropy_bf = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
        for itr in range(iters):
            edge = agt_edges[np.random.choice(range(len(agt_edges)))]
            q = np.where(edge != 0)[0]
            k, j = q[0], q[-1]
            nk, nj = k - self.world.n_adv, k - self.world.n_adv
            if k == j: # Agent is alone
                for _ in range(self.admm_iter):
                    self.train_n[k](*args_n[k], 0, nj, lr)
            else:
                for _ in range(self.admm_iter):
                    # print('The {}th iteration of adversaries!'.format(itr))
                    self.train_n[k](*args_n[k], edge[k], nj, lr)
                    self.train_n[j](*args_n[j], edge[j], nk, lr)

                a_k, p_k = self.to_exchange_n[k](obs_n[k], actions_n[k], cliprange, nj)
                a_j, p_j = self.to_exchange_n[j](obs_n[j], actions_n[j], cliprange, nk)
                self.exchange_n[k](nj, a_k, a_j, p_k, p_j, edge[j], obs_n[k], actions_n[k], cliprange, edge[k])
                self.exchange_n[j](nk, a_j, a_k, p_j, p_k, edge[k], obs_n[j], actions_n[j], cliprange, edge[j])

            loss_itr, vfloss_itr, entropy_itr = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
            imp = np.array(loss_bf).ravel() - np.array(loss_itr).ravel()
            print('  Inner iteration {}: {}'.format(itr, imp))
            # print('    {}, {}, {}'.format(loss_itr, vfloss_itr, entropy_itr))

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
