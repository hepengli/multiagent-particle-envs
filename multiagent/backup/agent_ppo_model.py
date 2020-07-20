from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import os.path as osp
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common.policies import PolicyWithValue
from baselines.common.vec_env.vec_env import VecEnv
from baselines.common import colorize
from baselines.common.cg import cg
from baselines.common.distributions import make_pdtype

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

def constfn(val):
    def f(frac):
        return val * (1 - frac)
    return f

class AgentModel(tf.Module):
    def __init__(self, agent, network, nsteps, rho, ent_coef, vf_coef, max_grad_norm, seed, load_path, **network_kwargs):
        super(AgentModel, self).__init__(name='MAPPO2Model')
        set_global_seeds(seed)
        # Get state_space and action_space
        ob_space = agent.observation_space
        ac_space = agent.action_space


        if isinstance(network, str):
            network_type = network
            policy_network_fn = get_network_builder(network_type)(**network_kwargs)
            network = policy_network_fn(ob_space.shape)

        self.train_model = PolicyWithValue(ac_space, network)
        if MPI is not None:
          self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.train_model.trainable_variables)
        else:
          self.optimizer = tf.keras.optimizers.Adam()

        # if isinstance(network, str):
        #     network = get_network_builder(network)(**network_kwargs)
        # policy_network = network(ob_space.shape)
        # value_network = network(ob_space.shape)
        # self.train_model = pi = PolicyWithValue(ac_space, policy_network, value_network)
        # self.pi_var_list = policy_network.trainable_variables + list(pi.pdtype.trainable_variables)
        # self.vf_var_list = value_network.trainable_variables + pi.value_fc.trainable_variables

        # if MPI is not None:
        #     self.pi_optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.pi_var_list)
        #     self.vf_optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.vf_var_list)
        # else:
        #     self.pi_optimizer = tf.keras.optimizers.Adam()
        #     self.vf_optimizer = tf.keras.optimizers.Adam()
        self.agent = agent
        self.nsteps = nsteps
        self.rho = rho
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        self.loss_names = ['Lagrange_loss', 'sync_loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        if MPI is not None:
            sync_from_root(self.variables)

        self.comm_matrix = agent.comm_matrix.copy()
        self.estimates = np.ones([agent.nmates, nsteps], dtype=np.float32)
        self.multipliers = np.zeros([agent.nmates, nsteps], dtype=np.float32)
        for i, comm_i in enumerate(self.comm_matrix):
            self.estimates[i] = comm_i[self.agent.id] * self.estimates[i]

        if load_path is not None:
            load_path = osp.expanduser(load_path)
            ckpt = tf.train.Checkpoint(model=self.train_model)
            manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
            ckpt.restore(manager.latest_checkpoint)

    def reinitial_estimates(self):
        self.estimates = np.random.normal(0,0.1, [self.agent.nmates, self.nsteps]).astype(np.float32)
        self.multipliers = np.random.uniform(0,1,[self.agent.nmates, self.nsteps]).astype(np.float32)
        for i, comm_i in enumerate(self.comm_matrix):
            self.estimates[i] = comm_i[self.agent.id] * self.estimates[i]

    def store_oldpi_var(self):
        pi_var_list = self.train_model.policy_network.trainable_variables + \
                      list(self.train_model.pdtype.trainable_variables)
        self.oldpi_var_list = [var.numpy() for var in pi_var_list]

    def assign_new_eq_old(self):
        pi_var_list = self.train_model.policy_network.trainable_variables + \
                      list(self.train_model.pdtype.trainable_variables)
        for pi_var, old_pi_var in zip(pi_var_list, self.oldpi_var_list):
            pi_var.assign(old_pi_var)


    # @tf.function
    # def get_vf_grad(self, cliprange, obs, returns, actions, values, advs, neglogpac_old):
    #     with tf.GradientTape() as tape:
    #         vpred = self.train_model.value(obs)
    #         vpredclipped = values + tf.clip_by_value(vpred - values, -cliprange, cliprange)
    #         vf_losses1 = tf.square(vpred - returns)
    #         vf_losses2 = tf.square(vpredclipped - returns)
    #         vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    #     vf_grads = tape.gradient(vf_loss, self.vf_var_list)
    #     if self.max_grad_norm is not None:
    #         vf_grads, _ = tf.clip_by_global_norm(vf_grads, self.max_grad_norm)
    #     if MPI is not None:
    #         vf_grads = tf.concat([tf.reshape(g, (-1,)) for g in vf_grads], axis=0)

    #     return vf_grads, vf_loss


    @tf.function
    def get_pi_grad(self, cliprange, nb, estimates, multipliers, obs, returns, actions, values, advs, neglogpac_old):
        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, logits = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())

            vpred = self.train_model.value(obs)
            vpredclipped = values + tf.clip_by_value(vpred - values, -cliprange, cliprange)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            ratio = tf.exp(neglogpac_old - neglogpac)
            clipped_ratio = tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * clipped_ratio
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            comm = self.comm_matrix[self.comm_matrix[:,nb]!=0][0,self.agent.id]
            syncerr = comm * ratio - estimates
            sync_loss = tf.reduce_mean(multipliers * syncerr) + \
                        0.5 * self.rho * (tf.reduce_mean(tf.square(syncerr)))

            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - neglogpac_old))
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))

            loss = pg_loss + sync_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = self.train_model.trainable_variables
        grads = tape.gradient(loss, var_list)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        if MPI is not None:
            grads = tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)
        return grads, loss, pg_loss, sync_loss, vf_loss, entropy, approxkl, clipfrac

        # pi_grads = tape.gradient(pi_loss, self.pi_var_list)
        # if self.max_grad_norm is not None:
        #     pi_grads, _ = tf.clip_by_global_norm(pi_grads, self.max_grad_norm)
        # if MPI is not None:
        #     pi_grads = tf.concat([tf.reshape(g, (-1,)) for g in pi_grads], axis=0)
        # return pi_grads, pi_loss, pg_loss, sync_loss, entropy, approxkl, clipfrac


    def pi_update(self, lr, cliprange, nb, obs, returns, actions, values, advs, neglogpacs_old):
        estimates = self.estimates[nb]
        multipliers = self.multipliers[nb]
        pi_grads, pi_loss, pg_loss, sync_loss, vf_loss, entropy, approxkl, clipfrac = self.get_pi_grad(
            cliprange, nb, estimates, multipliers, obs, returns, actions, values, advs, neglogpacs_old)

        if MPI is not None:
            self.optimizer.apply_gradients(pi_grads, lr)
        else:
            self.optimizer.learning_rate = lr
            grads_and_vars = zip(pi_grads, self.train_model.trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)

        return pi_loss, pg_loss, sync_loss, vf_loss, entropy, approxkl, clipfrac

        # if MPI is not None:
        #     self.pi_optimizer.apply_gradients(pi_grads, lr)
        # else:
        #     self.pi_optimizer.learning_rate = lr
        #     grads_and_vars = zip(pi_grads, self.pi_var_list)
        #     self.pi_optimizer.apply_gradients(grads_and_vars)

        # return pi_loss, pg_loss, sync_loss, entropy, approxkl, clipfrac


    # def vf_update(self, lr, cliprange, obs, returns, actions, values, advs, neglogpacs_old):
    #     vf_grads, vf_loss = self.get_vf_grad(
    #         cliprange, obs, returns, actions, values, advs, neglogpacs_old)
    #     if MPI is not None:
    #         self.vf_optimizer.apply_gradients(vf_grads, lr)
    #     else:
    #         self.vf_optimizer.learning_rate = lr
    #         grads_and_vars = zip(vf_grads, self.train_model.trainable_variables)
    #         self.vf_optimizer.apply_gradients(grads_and_vars)

    #     return vf_loss


    def info_to_exchange(self, cliprange, ob, ac, neglogpac_old, nb):
        policy_latent = self.train_model.policy_network(ob)
        pd, logits = self.train_model.pdtype.pdfromlatent(policy_latent)
        neglogpac = pd.neglogp(ac)
        ratio = tf.exp(neglogpac_old - neglogpac)
        clipped_ratio = tf.clip_by_value(tf.exp(-neglogpac), 1-cliprange, 1+cliprange)

        return ratio, self.multipliers[nb]


    def exchange(self, cliprange, ob, ac, neglogpac_old, nb_ratio, nb_multipliers, nb):
        policy_latent = self.train_model.policy_network(ob)
        pd, logits = self.train_model.pdtype.pdfromlatent(policy_latent)
        neglogpac = pd.neglogp(ac)
        ratio = tf.exp(neglogpac_old - neglogpac)
        clipped_ratio = tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
        comm = self.comm_matrix[self.comm_matrix[:,nb]!=0][0,self.agent.id]

        v = 0.5 * (self.multipliers[nb] + nb_multipliers) + \
            0.5 * self.rho * (comm * ratio + (-comm) * nb_ratio)
        estimate = np.array((1.0/self.rho) * (self.multipliers[nb] - v) + comm * ratio)

        self.estimates = tf.tensor_scatter_nd_update(self.estimates, [[nb]], estimate[None,:])
        self.multipliers = tf.tensor_scatter_nd_update(self.multipliers, [[nb]], v[None,:])


    # @tf.function
    # def compute_klgrad(self, flat_tangents, ob, ac):
    #     with tf.GradientTape() as tape:
    #         old_policy_latent = self.oldpi.policy_network(ob)
    #         old_pd, _ = self.pi.pdtype.pdfromlatent(old_policy_latent)
    #         policy_latent = self.pi.policy_network(ob)
    #         pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
    #         kloldnew = old_pd.kl(pd)
    #         meankl = tf.reduce_mean(kloldnew)
    #     klgrad = tape.gradient(meankl, policy_latent)

    #     with tf.GradientTape() as tape:
    #         old_policy_latent = self.oldpi.policy_network(ob)
    #         policy_latent = self.pi.policy_network(ob)

    #         a0_old = old_policy_latent[:,:3] - tf.reduce_max(old_policy_latent[:,:3], axis=-1, keepdims=True)
    #         a0_new = policy_latent[:,:3] - tf.reduce_max(policy_latent[:,:3], axis=-1, keepdims=True)
    #         a1_old = old_policy_latent[:,3:] - tf.reduce_max(old_policy_latent[:,3:], axis=-1, keepdims=True)
    #         a1_new = policy_latent[:,3:] - tf.reduce_max(policy_latent[:,3:], axis=-1, keepdims=True)
    #         ea0_old, ea1_old = tf.exp(a0_old), tf.exp(a1_old)
    #         ea0_new, ea1_new = tf.exp(a0_new), tf.exp(a1_new)
    #         z0_old, z1_old = tf.reduce_sum(ea0_old, axis=-1, keepdims=True), tf.reduce_sum(ea1_old, axis=-1, keepdims=True)
    #         z0_new, z1_new = tf.reduce_sum(ea0_new, axis=-1, keepdims=True), tf.reduce_sum(ea1_new, axis=-1, keepdims=True)
    #         p0_old, p1_old = ea0_old / z0_old, ea1_old / z1_old
    #         p0_new, p1_new = ea0_new / z0_new, ea1_new / z1_new

    #         kl0 = tf.reduce_sum(p0_old * tf.math.log(p0_old) - p0_old * tf.math.log(p0_new), axis=1)
    #         kl1 = tf.reduce_sum(p1_old * tf.math.log(p1_old) - p1_old * tf.math.log(p1_new), axis=1)

    #         meankl = tf.reduce_mean(kl0+kl1)
    #     klgrad_1 = tape.gradient(meankl, p0_new)

    #     return klgrad, klgrad_1, policy_latent, old_policy_latent

    # @tf.function
    # def compute_jaco(self, flat_tangents, ob, ac):

    #     with tf.GradientTape() as tape:
    #         old_policy_latent = self.oldpi.policy_network(ob)
    #         old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
    #         policy_latent = self.pi.policy_network(ob)
    #         pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
    #         ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
    #     jaco = tape.jacobian(ratio, policy_latent)

    #     with tf.GradientTape() as tape:
    #         old_policy_latent = self.oldpi.policy_network(ob)
    #         old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
    #         policy_latent = self.pi.policy_network(ob)
    #         pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
    #         prob = - tf.reduce_mean(flat_tangents*pd.logp(ac))
    #     grad = tape.jacobian(prob, policy_latent)

    #     return jaco, grad

#                 var_list = pi.net.w
#                 klgrads = tf.gradients(dist, var_list)
#                 print(klgrads)
#                 input()
#                 flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")

#                 shapes = [var.get_shape().as_list() for var in var_list]
#                 start = 0
#                 tangents = []
#                 for shape in shapes:
#                     sz = U.intprod(shape)
#                     tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
#                     start += sz

#                 jjvp = [tf.zeros(shape, dtype=tf.float32) for shape in shapes]
#                 jtvp = [tf.zeros(shape, dtype=tf.float32) for shape in shapes]
#                 right_b = - ADV + A * tf.gather(pi.net.p, NB) - rho * A * tf.gather(pi.net.z, NB)
#                 for i in range(self.n_batches):
#                     ratio_i_grad = tf.gradients(ratio[i], var_list)
#                     jvp_i = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(ratio_i_grad, tangents)])
#                     jjvp = [tf.add_n([jj, gg*jvp_i]) for (jj, gg) in zipsame(jjvp, ratio_i_grad)]
#                     jtvp = [tf.add_n([jt, gt*right_b[i]]) for (jt, gt) in zipsame(jtvp, ratio_i_grad)]

#                 jjvp = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in jjvp])
#                 jtvp = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in jtvp])
#                 gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
#                 fvp = tf.add_n([U.flatgrad(gvp, var_list), rho * jjvp])

#                 # Define the value loss
#                 vpredclipped = OLDVPRED + tf.clip_by_value(pi.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
#                 # vpredclipped = tf.clip_by_value(pi.vf, OLDVPRED*(1-CLIPRANGE), OLDVPRED*(1+CLIPRANGE))
#                 vferr = tf.square(pi.vf - R)
#                 vferr2 = tf.square(vpredclipped - R)
#                 vf_loss = .5 * tf.reduce_mean(tf.maximum(vferr, vferr2))
#                 vfadam = MpiAdam(pi.net.v)

#                 compute_jtvp = U.function([OB, AC, ADV, A, NB], jtvp)
#                 compute_fvp = U.function([flat_tangent, OB, AC, ADV], fvp)
#                 compute_losses = U.function([OB, AC, ADV, A, NB], losses)
#                 compute_vfloss = U.function([OB, R, OLDVPRED, CLIPRANGE], vf_loss)
#                 exchange = pi.net.exchange(sess, OB, AC, CLIPRANGE, NB, rho)
#                 to_exchange = U.function([OB, AC, ADV, NB, CLIPRANGE], [ratio, tf.gather(pi.net.p, NB)])

#                 get_flat = U.GetFlat(var_list)
#                 set_from_flat = U.SetFromFlat(var_list)

#             self.pi_n.append(pi)
#             self.oldpi_n.append(oldpi)
#             self.get_flat_n.append(get_flat)
#             self.set_from_flat_n.append(set_from_flat)
#             self.vfadam_n.append(vfadam)
#             self.exchange_n.append(exchange)
#             self.to_exchange_n.append(to_exchange)
#             self.compute_jtvp_n.append(compute_jtvp)
#             self.compute_fvp_n.append(compute_fvp)
#             self.compute_losses_n.append(compute_losses)
#             self.compute_vfloss_n.append(compute_vfloss)

#         # Update old plicy network
#         updates = []
#         for i in range(len(world.agents)):
#             name_scope = world.agents[i].name.replace(' ', '')
#             old_vars = get_trainable_variables("{}/oldpi".format(name_scope))
#             now_vars = get_trainable_variables("{}/pi".format(name_scope))
#             updates += [tf.assign(oldv, nowv) for (oldv, nowv) in zipsame(old_vars, now_vars)]
#             updates += [tf.assign(self.pi_n[i].net.z, tf.ones_like(self.pi_n[i].net.z))]
#         self.assign_old_eq_new = U.function([],[], updates=updates)

#         @contextmanager
#         def timed(msg):
#             print(colorize(msg, color='magenta'))
#             tstart = time.time()
#             yield
#             print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
#         self.timed = timed

#         def allmean(x):
#             assert isinstance(x, np.ndarray)
#             if MPI is not None:
#                 out = np.empty_like(x)
#                 MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
#                 out /= self.nworkers
#             else:
#                 out = np.copy(x)

#             return out
#         self.allmean = allmean

#         # Initialization
#         U.initialize()
#         if load_path is not None:
#             self.load(load_path)

#         # for i in range(len(self.pi_n)):
#             th_init = self.get_flat_n[i]()
#             self.set_from_flat_n[i](th_init)
#             print("Init param sum", th_init.sum(), flush=True)

#         for vfadam in self.vfadam_n:
#             vfadam.sync()

#     def load(self, load_path):
#         # variables = []
#         # for agent in self.world.agents:
#         #     name_scope = agent.name.replace(' ', '')
#         #     variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
#         # U.load_variables(load_path, variables=variables, sess=self.sess)
#         for pi in self.pi_n:
#             pi.load(load_path)

#     def save(self, save_path):
#         # variables = []
#         # for agent in self.world.agents:
#         #     name_scope = agent.name.replace(' ', '')
#         #     variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope))
#         # U.save_variables(save_path, variables=variables, sess=self.sess)
#         for pi in self.pi_n:
#             pi.save(save_path)

#     def print_variables(self):
#         for i, pi in enumerate(self.pi_n):
#             w = self.sess.run(pi.net.w)
#             print('------------------------------')
#             print('Agent {}:'.format(i))
#             print(w)


#     # def step(self, obs_n, nenv=1):
#     #     if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
#     #     actions_n, values_n = [], []
#     #     for i in range(self.world.n):
#     #         actions, values = self.pi_n[i].step(obs_n[i])
#     #         actions_n.append(actions)
#     #         values_n.append(values)
#     #     # print(actions_n, values_n)
#     #     actions_n = np.hstack(actions_n)
#     #     values_n = np.hstack(values_n)
#     #     # print(actions_n, values_n)
#     #     # input()
#     #     if nenv == 1: actions_n = actions_n.squeeze()

#     #     return actions_n, values_n

#     # def value(self, obs_n, nenv):
#     #     if nenv > 1: obs_n = [np.asarray(ob) for ob in zip(*obs_n)]
#     #     return np.hstack([self.pi_n[i].value(obs_n[i]) for i in range(self.world.n)])

#     # def share_actions(self, actions_n):
#     #     shared_actions_n = list(actions_n)
#     #     for i in range(self.world.n_adv):
#     #         shared_actions_n[i] = np.hstack(actions_n[:self.world.n_adv]).astype(actions_n[i].dtype).squeeze()
#     #     for i in range(self.world.n_adv, self.world.n):
#     #         shared_actions_n[i] = np.hstack(actions_n[self.world.n_adv:]).astype(actions_n[i].dtype).squeeze()

#     #     return shared_actions_n
#     def step(self, obs_n):
#         return zip(*[self.pi_n[i].step(obs_n[i]) for i in range(self.env.n)])

#     def value(self, obs_n):
#         return [self.pi_n[i].value(obs_n[i]) for i in range(self.env.n)]

#     def share_actions(self, actions_n):
#         shared_actions_n = list(actions_n)
#         if hasattr(self.env.agents[0], 'adversary'):
#             n_a = self.env.world.num_adversaries
#             n_g = self.env.world.num_good_agents
#         else:
#             n_a = len(self.env.agents)
#             n_g= 0

#         for i in range(n_a):
#             shared_actions_n[i] = np.hstack(actions_n[:n_a]).astype(actions_n[i].dtype).squeeze()
#         for i in range(n_g):
#             shared_actions_n[n_a+i] = np.hstack(actions_n[n_a:]).astype(actions_n[n_a+i].dtype).squeeze()

#         return shared_actions_n

#     def train_adv(self, iters, cliprange, lr, obs_n, actions_n, values_n, advs_n, returns_n, dones_n):
#         # some constants
#         eps = 1e-8
#         A = self.world.A
#         adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
#         agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

#         for i, pi in enumerate(self.pi_n):
#             if hasattr(pi, "ret_rms"): pi.ret_rms.update(returns_n[i])
#             if hasattr(pi, "ob_rms"): pi.ob_rms.update(obs_n[i]) # update running mean/std for policy

#         # Set old parameter values to new parameter values
#         self.assign_old_eq_new()

#         # Prepare data
#         advs_n = [(advs_n[i] - advs_n[i].mean()) / (advs_n[i].std() + eps) for i in range(self.world.n_adv)]
#         for i in range(self.world.n_adv):
#             obs_n[i] = np.squeeze(obs_n[i])
#             actions_n[i] = np.squeeze(actions_n[i])
#             values_n[i] = np.squeeze(values_n[i])
#             advs_n[i] = np.squeeze(advs_n[i])
#             returns_n[i] = np.squeeze(returns_n[i])
#         # for i in range(self.world.n_adv):
#         #     print(obs_n[i].shape, actions_n[i].shape, values_n[i].shape, advs_n[i].shape, returns_n[i].shape)
#         #     input()

#         # Train adversaries
#         for itr in range(iters):
#             edge = adv_edges[np.random.choice(range(len(adv_edges)))]
#             q = np.where(edge != 0)[0]
#             k, j = q[0], q[-1]
#             if k == j: # Agent is alone
#                 # for _ in range(self.admm_iter):
#                 #     self.train_n[k](*args_n[k], 0, j, lr)
#                 pass
#             else:
#                 def fisher_vector_product(p):
#                     return self.allmean(self.compute_fvp_n[k](p, obs_n[k], actions_n[k], advs_n[k])) + 0.02 * p
#                 g_k = self.compute_jtvp_n[k](obs_n[k], actions_n[k], advs_n[k], edge[k], j)
#                 g_k = self.allmean(g_k)
#                 if np.allclose(g_k, 0):
#                     logger.log("Got zero gradient. not updating")
#                 else:
#                     with self.timed("cg"):
#                         stepdir = cg(fisher_vector_product, g_k, cg_iters=10, verbose=self.rank==0)
#                     assert np.isfinite(stepdir).all()
#                     shs = .5*stepdir.dot(fisher_vector_product(stepdir))
#                     lm = np.sqrt(shs / self.max_kl)
#                     # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
#                     fullstep = stepdir / lm
#                     expectedimprove = g_k.dot(fullstep)

#                     loss_bf, surr_bf, kl_bf = self.allmean(np.array(self.compute_losses_n[i](obs_n[k], actions_n[k], advs_n[k], edge[k], j)))
#                     stepsize = 1.0
#                     thbefore = self.get_flat_n[k]()
#                     for _ in range(10):
#                         thnew = thbefore - fullstep * stepsize
#                         self.set_from_flat_n[k](thnew)
#                         meanlosses = loss, surr, kl = self.allmean(np.array(self.compute_losses_n[i](obs_n[k], actions_n[k], advs_n[k], edge[k], j)))
#                         improve = loss_bf - loss
#                         logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
#                         if not np.isfinite(meanlosses).all():
#                             logger.log("Got non-finite value of losses -- bad!")
#                         elif kl > self.max_kl * 1.5:
#                             logger.log("violated KL constraint. shrinking step.")
#                         elif improve < 0:
#                             logger.log("surrogate didn't improve. shrinking step.")
#                         else:
#                             logger.log("Stepsize OK!")
#                             break
#                         stepsize *= .5
#                     else:
#                         logger.log("couldn't compute a good step")
#                         self.set_from_flat_n[k](thbefore)

#                 def fisher_vector_product(p):
#                     return self.allmean(self.compute_fvp_n[j](p, obs_n[j], actions_n[j], advs_n[j])) + 0.02 * p
#                 g_j = self.compute_jtvp_n[j](obs_n[j], actions_n[j], advs_n[j], edge[j], k)
#                 g_j = self.allmean(g_j)
#                 if np.allclose(g_j, 0):
#                     logger.log("Got zero gradient. not updating")
#                 else:
#                     with self.timed("cg"):
#                         stepdir = cg(fisher_vector_product, g_j, cg_iters=10, verbose=self.rank==0)
#                     assert np.isfinite(stepdir).all()
#                     shs = .5*stepdir.dot(fisher_vector_product(stepdir))
#                     lm = np.sqrt(shs / self.max_kl)
#                     # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
#                     fullstep = stepdir / lm
#                     expectedimprove = g_j.dot(fullstep)

#                     loss_bf, surr_bf, kl_bf = self.allmean(np.array(self.compute_losses_n[i](obs_n[j], actions_n[j], advs_n[j], edge[j], k)))
#                     stepsize = 1.0
#                     thbefore = self.get_flat_n[j]()
#                     for _ in range(10):
#                         thnew = thbefore - fullstep * stepsize
#                         self.set_from_flat_n[j](thnew)
#                         meanlosses = loss, surr, kl = self.allmean(np.array(self.compute_losses_n[i](obs_n[j], actions_n[j], advs_n[j], edge[j], k)))
#                         improve = loss_bf - loss
#                         logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
#                         if not np.isfinite(meanlosses).all():
#                             logger.log("Got non-finite value of losses -- bad!")
#                         elif kl > self.max_kl * 1.5:
#                             logger.log("violated KL constraint. shrinking step.")
#                         elif improve < 0:
#                             logger.log("surrogate didn't improve. shrinking step.")
#                         else:
#                             logger.log("Stepsize OK!")
#                             break
#                         stepsize *= .5
#                     else:
#                         logger.log("couldn't compute a good step")
#                         self.set_from_flat_n[j](thbefore)

#                 if self.sync:
#                     a_k, p_k = self.to_exchange_n[k](obs_n[k], actions_n[k], advs_n[k], cliprange, j)
#                     a_j, p_j = self.to_exchange_n[j](obs_n[j], actions_n[j], advs_n[j], cliprange, k)
#                     self.exchange_n[k](j, a_k, a_j, p_k, p_j, edge[j], obs_n[k], actions_n[k], cliprange, edge[k])
#                     self.exchange_n[j](k, a_j, a_k, p_j, p_k, edge[k], obs_n[j], actions_n[j], cliprange, edge[j])

#                     z_k = self.sess.run(self.pi_n[k].net.z[j])
#                     z_j = self.sess.run(self.pi_n[j].net.z[k])
#                     print('zk + zj: {}'.format(np.sum(np.square(z_k+z_j))))

#             print('  Inner iteration {}'.format(itr))

#     def train_agt(self, iters, cliprange, lr, obs_n, actions_n, mb_rewards, values_n, advs_n, returns_n, dones_n):
#         # some constants
#         eps = 1e-8
#         A = self.world.A
#         adv_edges = A[np.unique(np.nonzero(A[:,:self.world.n_adv])[0])]
#         agt_edges = A[np.unique(np.nonzero(A[:,self.world.n_adv:])[0])]

#         # Set old parameter values to new parameter values
#         self.assign_old_eq_new()

#         # Prepare data
#         advs_n = [(advs_n[i] - advs_n[i].mean()) / (advs_n[i].std() + eps) for i in range(self.world.n)]
#         args_n = [(obs_n[i], actions_n[i], advs_n[i], returns_n[i], values_n[i], cliprange) for i in range(self.world.n)]

#         # Train good agents
#         loss_bf, vfloss_bf, entropy_bf = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
#         for itr in range(iters):
#             edge = agt_edges[np.random.choice(range(len(agt_edges)))]
#             q = np.where(edge != 0)[0]
#             k, j = q[0], q[-1]
#             nk, nj = k - self.world.n_adv, k - self.world.n_adv
#             if k == j: # Agent is alone
#                 for _ in range(self.admm_iter):
#                     self.train_n[k](*args_n[k], 0, nj, lr)
#             else:
#                 for _ in range(self.admm_iter):
#                     # print('The {}th iteration of adversaries!'.format(itr))
#                     self.train_n[k](*args_n[k], edge[k], nj, lr)
#                     self.train_n[j](*args_n[j], edge[j], nk, lr)

#                 a_k, p_k = self.to_exchange_n[k](obs_n[k], actions_n[k], cliprange, nj)
#                 a_j, p_j = self.to_exchange_n[j](obs_n[j], actions_n[j], cliprange, nk)
#                 self.exchange_n[k](nj, a_k, a_j, p_k, p_j, edge[j], obs_n[k], actions_n[k], cliprange, edge[k])
#                 self.exchange_n[j](nk, a_j, a_k, p_j, p_k, edge[k], obs_n[j], actions_n[j], cliprange, edge[j])

#             loss_itr, vfloss_itr, entropy_itr = list(zip(*[self.losses_n[i](*args_n[i]) for i in range(self.world.n_adv, self.world.n)]))
#             imp = np.array(loss_bf).ravel() - np.array(loss_itr).ravel()
#             print('  Inner iteration {}: {}'.format(itr, imp))
#             # print('    {}, {}, {}'.format(loss_itr, vfloss_itr, entropy_itr))

# def get_variables(scope):
#     return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

# def get_trainable_variables(scope):
#     return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

# def get_vf_trainable_variables(scope):
#     return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

# def get_pi_trainable_variables(scope):
#     return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
