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

from contextlib import contextmanager
MPI = None

class AgentModel(tf.Module):
    def __init__(self, agent, network, nsteps, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
                 cg_damping, cg_iters, seed, load_path, **network_kwargs):
        super(AgentModel, self).__init__(name='MATRPOModel')
        self.agent = agent
        self.nsteps = nsteps
        self.rho = rho
        self.max_kl = max_kl
        self.ent_coef = ent_coef
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.vf_stepsize = vf_stepsize
        self.vf_iters = vf_iters

        set_global_seeds(seed)

        np.set_printoptions(precision=3)

        if MPI is not None:
            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.nworkers = 1
            self.rank = 0

        # Setup losses and stuff
        # ----------------------------------------
        ob_space = agent.observation_space
        ac_space = agent.action_space

        if isinstance(network, str):
            network = get_network_builder(network)(**network_kwargs)

        with tf.name_scope(agent.name):
            with tf.name_scope("pi"):
                pi_policy_network = network(ob_space.shape)
                pi_value_network = network(ob_space.shape)
                self.pi = pi = PolicyWithValue(ac_space, pi_policy_network, pi_value_network)
            with tf.name_scope("oldpi"):
                old_pi_policy_network = network(ob_space.shape)
                old_pi_value_network = network(ob_space.shape)
                self.oldpi = oldpi = PolicyWithValue(ac_space, old_pi_policy_network, old_pi_value_network)

        self.comm_matrix = agent.comm_matrix.copy()
        self.estimates = np.random.uniform(0, 1, [agent.nmates, nsteps, self.agent.action_space.nvec.sum()]).astype(np.float32)
        self.multipliers = np.random.uniform(0, 1, [self.agent.nmates, self.nsteps, self.agent.action_space.nvec.sum()]).astype(np.float32)
        for i, comm_i in enumerate(self.comm_matrix):
            self.estimates[i] = comm_i[self.agent.id] * self.estimates[i]

        pi_var_list = pi_policy_network.trainable_variables + list(pi.pdtype.trainable_variables)
        old_pi_var_list = old_pi_policy_network.trainable_variables + list(oldpi.pdtype.trainable_variables)
        vf_var_list = pi_value_network.trainable_variables + pi.value_fc.trainable_variables
        old_vf_var_list = old_pi_value_network.trainable_variables + oldpi.value_fc.trainable_variables

        self.pi_var_list = pi_var_list
        self.old_pi_var_list = old_pi_var_list
        self.vf_var_list = vf_var_list
        self.old_vf_var_list = old_vf_var_list

        if load_path is not None:
            load_path = osp.expanduser(load_path)
            ckpt = tf.train.Checkpoint(model=pi)
            manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
            ckpt.restore(manager.latest_checkpoint)

        self.vfadam = MpiAdam(vf_var_list)

        self.get_flat = U.GetFlat(pi_var_list)
        self.set_from_flat = U.SetFromFlat(pi_var_list)
        self.loss_names = ["Lagrange", "surrgain", "sync", "meankl", "entloss", "entropy"]
        self.shapes = [var.get_shape().as_list() for var in pi_var_list]

    def reinitial_estimates(self):
        self.estimates = np.random.uniform(0, 1, [self.agent.nmates, self.nsteps, self.agent.action_space.nvec.sum()]).astype(np.float32)
        self.multipliers = np.random.uniform(0, 1, [self.agent.nmates, self.nsteps, self.agent.action_space.nvec.sum()]).astype(np.float32)
        for i, comm_i in enumerate(self.comm_matrix):
            self.estimates[i] = comm_i[self.agent.id] * self.estimates[i]

    def assign_old_eq_new(self):
        for pi_var, old_pi_var in zip(self.pi_var_list, self.old_pi_var_list):
            old_pi_var.assign(pi_var)
        for vf_var, old_vf_var in zip(self.vf_var_list, self.old_vf_var_list):
            old_vf_var.assign(vf_var)

    def assign_new_eq_old(self):
        for old_pi_var, pi_var in zip(self.old_pi_var_list, self.pi_var_list):
            pi_var.assign(old_pi_var)
        for old_vf_var, vf_var in zip(self.old_vf_var_list, self.vf_var_list):
            vf_var.assign(old_vf_var)

    @contextmanager
    def timed(self, msg):
        if self.rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= self.nworkers
        else:
            out = np.copy(x)

        return out

    @tf.function
    def compute_losses(self, ob, ac, atarg, estimates, multipliers, nb):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, old_logits = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        ent = pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.ent_coef * meanent
        ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        surrgain = tf.reduce_mean(ratio * atarg)

        nvec = self.agent.action_space.nvec
        old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
        old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
        old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [5], axis=1)
        old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
        p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
        p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
        p1 = p0p1 - tf.repeat(p0[:,0:1], [5], axis=1)
        p = tf.exp(tf.concat([p0, p1], axis=1))

        comm = self.comm_matrix[self.comm_matrix[:,nb]!=0][0,self.agent.id]
        syncerr = comm * (p / tf.sqrt(old_p)+1e-10) - estimates
        syncloss = tf.reduce_mean(tf.reduce_mean(multipliers * syncerr, axis=-1)) + \
                   0.5 * self.rho * tf.reduce_mean(tf.reduce_sum(tf.square(syncerr), axis=1), axis=0)
        lagrangeloss = - surrgain - entbonus + syncloss
        losses = [lagrangeloss, surrgain, syncloss, meankl, entbonus, meanent]
        return losses


    #ob shape should be [batch_size, ob_dim], merged nenv
    #ret shape should be [batch_size]
    @tf.function
    def compute_vflossandgrad(self, ob, ret):
        with tf.GradientTape() as tape:
            pi_vf = self.pi.value(ob)
            vferr = tf.reduce_mean(tf.square(pi_vf - ret))
        return U.flatgrad(tape.gradient(vferr, self.vf_var_list), self.vf_var_list)


    @tf.function
    def compute_vjp(self, ob, ac, atarg, estimates, multipliers, nb):
        with tf.GradientTape() as tape:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
            surrgain = ratio * atarg

            nvec = self.agent.action_space.nvec
            old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
            old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
            old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [5], axis=1)
            old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
            p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
            p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
            p1 = p0p1 - tf.repeat(p0[:,0:1], [5], axis=1)
            p = tf.exp(tf.concat([p0, p1], axis=1))

            comm = self.comm_matrix[self.comm_matrix[:,nb]!=0][0,self.agent.id]
            v = - comm * multipliers + self.rho * comm * estimates
            vpr = tf.reduce_mean(surrgain + tf.reduce_sum(v * p / (tf.sqrt(old_p)+1e-10), axis=1))
        vjp = tape.jacobian(vpr, self.pi_var_list)

        return U.flatgrad(vjp, self.pi_var_list)

    @tf.function
    def my_compute_fvp(self, tangents, ob, ac):
        with tf.autodiff.ForwardAccumulator(
            primals=self.pi_var_list,
            # The "vector" in Hessian-vector product.
            tangents=tangents) as acc:
            with tf.GradientTape() as tape:
                old_policy_latent = self.oldpi.policy_network(ob)
                old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
                policy_latent = self.pi.policy_network(ob)
                pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
                kloldnew = old_pd.kl(pd)
                meankl = (1. + self.rho) * tf.reduce_mean(kloldnew)
            backward = tape.gradient(meankl, self.pi_var_list)
        fvp = acc.jvp(backward)

        return U.flatgrad(fvp, self.pi_var_list)

    @tf.function
    def compute_jjvp(self, tangents, ob, ac):
        with tf.autodiff.ForwardAccumulator(
                primals=self.pi_var_list,
                tangents=tangents) as acc:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)

            nvec = self.agent.action_space.nvec
            old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
            old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
            old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [nvec[1]], axis=1)
            old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
            p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
            p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
            p1 = p0p1 - tf.repeat(p0[:,0:1], [nvec[1]], axis=1)
            p = tf.exp(tf.concat([p0, p1], axis=1))
            ratio = p / (tf.sqrt(old_p)+1e-10)
        jvp = acc.jvp(ratio)

        with tf.GradientTape() as tape:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)

            nvec = self.agent.action_space.nvec
            old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
            old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
            old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [nvec[1]], axis=1)
            old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
            p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
            p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
            p1 = p0p1 - tf.repeat(p0[:,0:1], [nvec[1]], axis=1)
            p = tf.exp(tf.concat([p0, p1], axis=1))
            ratio = p / (tf.sqrt(old_p)+1e-10)

            jvpr = self.rho * tf.reduce_mean(tf.reduce_sum(jvp * ratio, axis=1), axis=0)
        jjvp = tape.jacobian(jvpr, self.pi_var_list)

        return U.flatgrad(jjvp, self.pi_var_list)


    @tf.function
    def compute_hvp(self, flat_tangents, ob, ac):
        tangents = self.reshape_from_flat(flat_tangents)
        fvp = self.my_compute_fvp(tangents, ob, ac)
        jjvp = self.compute_jjvp(tangents, ob, ac)

        return fvp + jjvp

    def reshape_from_flat(self, flat_tangents):
        shapes = [var.get_shape().as_list() for var in self.pi_var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangents[start:start+sz], shape))
            start += sz
        
        return tangents

    def info_to_exchange(self, ob, ac, nb):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        # ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))

        nvec = self.agent.action_space.nvec
        old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
        old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
        old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [5], axis=1)
        old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
        p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
        p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
        p1 = p0p1 - tf.repeat(p0[:,0:1], [5], axis=1)
        p = tf.exp(tf.concat([p0, p1], axis=1))
        ratio = p / (tf.sqrt(old_p) + 1e-10)

        return ratio, self.multipliers[nb]

    def exchange(self, ob, ac, nb_ratio, nb_multipliers, nb):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        # ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        comm = self.comm_matrix[self.comm_matrix[:,nb]!=0][0,self.agent.id]

        nvec = self.agent.action_space.nvec
        old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
        old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
        old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [5], axis=1)
        old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
        p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
        p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
        p1 = p0p1 - tf.repeat(p0[:,0:1], [5], axis=1)
        p = tf.exp(tf.concat([p0, p1], axis=1))
        ratio = p / (tf.sqrt(old_p) + 1e-10)

        v = 0.5 * (self.multipliers[nb] + nb_multipliers) + \
            0.5 * self.rho * (comm * ratio + (-comm) * nb_ratio)
        self.estimates[nb] = np.array((1.0/self.rho) * (self.multipliers[nb] - v) + comm * ratio).copy()
        self.multipliers[nb] = np.array(v).copy()

    def update(self, obs, actions, atarg, returns, vpredbefore, nb):
        obs = tf.constant(obs)
        actions = tf.constant(actions)
        atarg = tf.constant(atarg)
        returns = tf.constant(returns)
        estimates = tf.constant(self.estimates[nb])
        multipliers = tf.constant(self.multipliers[nb])
        args = obs, actions, atarg, estimates, multipliers
        # Sampling every 5
        fvpargs = [arr[::1] for arr in (obs, actions)]

        hvp = lambda p: self.allmean(self.compute_hvp(p, *fvpargs).numpy()) + self.cg_damping * p
        jjvp = lambda p: self.allmean(self.compute_jjvp(self.reshape_from_flat(p), *fvpargs).numpy()) + self.cg_damping * p
        fvp = lambda p: self.allmean(self.my_compute_fvp(self.reshape_from_flat(p), *fvpargs).numpy()) + self.cg_damping * p
        self.assign_new_eq_old() # set old parameter values to new parameter values


        # with self.timed("computegrad"):
        lossbefore = self.compute_losses(*args, nb)
        g = self.compute_vjp(*args, nb)
        lossbefore = self.allmean(np.array(lossbefore))
        g = g.numpy()
        g = self.allmean(g)

        # # check
        # v1 = jjvp(g)
        # v2 = fvp(g)
        # print(v1)
        # print(v2)
        # input()

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            # with self.timed("cg"):
            stepdir = cg(fvp, g, cg_iters=self.cg_iters, verbose=self.rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fvp(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            lagrangebefore, surrbefore, syncbefore, *_ = lossbefore
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = lagrange, surr, syncloss, kl, *_ = self.allmean(np.array(self.compute_losses(*args, nb)))
                improve = lagrangebefore - lagrange
                performance_improve = surr - surrbefore
                sync_improve = syncbefore - syncloss
                print(lagrangebefore, surrbefore, syncbefore)
                print(lagrange, surr, syncloss)
                # input()
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
                self.set_from_flat(thbefore)

        # with self.timed("vf"):
        for _ in range(self.vf_iters):
            for (mbob, mbret) in dataset.iterbatches((obs, returns),
            include_final_partial_batch=False, batch_size=64):
                vg = self.allmean(self.compute_vflossandgrad(mbob, mbret).numpy())
                self.vfadam.update(vg, self.vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, returns))




"""

    @tf.function
    def check_compute_vjp(self, ob, ac, tangents):

policies[0].assign_old_eq_new()

ob = np.random.normal(0,1,[4,11])
ac =  None
atarg = None
tangents = policies[1].pi_var_list
flat_tangents = U.flatgrad(tangents, policies[1].pi_var_list)[:,None]


with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        old_policy_latent = policies[0].oldpi.policy_network(ob)
        old_pd, _ = policies[0].oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = policies[0].pi.policy_network(ob)
        pd, _ = policies[0].pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        meankl = tf.reduce_mean(kloldnew)
    backward = tape.gradient(meankl, policies[0].pi_var_list)
    flat_backward = U.flatgrad(backward, policies[0].pi_var_list)
hess = outer_tape.jacobian(flat_backward, policies[0].pi_var_list)

flat_hess = []
for v in hess:
    print(v.shape)
    if len(v.shape)>2:
        flat_hess.append(tf.squeeze(tf.reshape(v, [int(v.shape[0]), int(v.shape[1])*int(v.shape[2])])))
    else:
        flat_hess.append(tf.squeeze(v))
flat_hess = tf.concat(flat_hess, axis=1)
fvp = tf.squeeze(tf.matmul(flat_hess, flat_tangents))
policies[0].my_compute_fvp(tangents, ob, ac)



with tf.autodiff.ForwardAccumulator(
        primals=policies[0].pi_var_list,
        tangents=tangents) as acc:
    old_policy_latent = policies[0].oldpi.policy_network(ob)
    old_pd, _ = policies[0].oldpi.pdtype.pdfromlatent(old_policy_latent)
    policy_latent = policies[0].pi.policy_network(ob)
    pd, _ = policies[0].pi.pdtype.pdfromlatent(policy_latent)

    nvec = policies[0].agent.action_space.nvec
    old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
    old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
    old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [nvec[1]], axis=1)
    old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
    p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
    p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
    p1 = p0p1 - tf.repeat(p0[:,0:1], [nvec[1]], axis=1)
    p = tf.exp(tf.concat([p0, p1], axis=1))
    ratio = p / tf.sqrt(old_p)
jvp = acc.jvp(ratio)

with tf.GradientTape() as tape:
    old_policy_latent = policies[0].oldpi.policy_network(ob)
    old_pd, _ = policies[0].oldpi.pdtype.pdfromlatent(old_policy_latent)
    policy_latent = policies[0].pi.policy_network(ob)
    pd, _ = policies[0].pi.pdtype.pdfromlatent(policy_latent)

    nvec = policies[0].agent.action_space.nvec
    old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
    old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
    old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [nvec[1]], axis=1)
    old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
    p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
    p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
    p1 = p0p1 - tf.repeat(p0[:,0:1], [nvec[1]], axis=1)
    p = tf.exp(tf.concat([p0, p1], axis=1))
    ratio = p / tf.sqrt(old_p)

    jvpr = tf.reduce_mean(tf.reduce_sum(jvp * ratio, axis=1), axis=0)
jjvp = tape.jacobian(jvpr, policies[0].pi_var_list)
jjvp = U.flatgrad(jjvp, policies[0].pi_var_list)








# vjp
with tf.GradientTape() as tape:
    old_policy_latent = policies[0].oldpi.policy_network(ob)
    old_pd, _ = policies[0].oldpi.pdtype.pdfromlatent(old_policy_latent)
    policy_latent = policies[0].pi.policy_network(ob)
    pd, _ = policies[0].pi.pdtype.pdfromlatent(policy_latent)

    nvec = policies[0].agent.action_space.nvec
    old_p0 = tf.transpose(tf.stack([old_pd.logp([i]) for i in range(nvec[0])]))
    old_p0p1 = tf.transpose(tf.stack([old_pd.logp([0,i]) for i in range(nvec[1])]))
    old_p1 = old_p0p1 - tf.repeat(old_p0[:,0:1], [nvec[1]], axis=1)
    old_p = tf.exp(tf.concat([old_p0, old_p1], axis=1))
    p0 = tf.transpose(tf.stack([pd.logp([i]) for i in range(nvec[0])]))
    p0p1 = tf.transpose(tf.stack([pd.logp([0,i]) for i in range(nvec[1])]))
    p1 = p0p1 - tf.repeat(p0[:,0:1], [nvec[1]], axis=1)
    p = tf.exp(tf.concat([p0, p1], axis=1))
    ratio = p / tf.sqrt(old_p)
vjp = tape.jacobian(ratio, policies[0].pi_var_list)
print([v.shape for v in vjp])






flat_vjp = []
for v in vjp:
    print(v.shape)
    if len(v.shape)>3:
        flat_vjp.append(tf.reshape(v, [int(v.shape[0]), int(v.shape[1]), int(v.shape[2])*int(v.shape[3])]))
    else:
        flat_vjp.append(v)
print([v.shape for v in flat_vjp])
flat_vjp = tf.concat(flat_vjp, axis=2)

h = tf.reduce_mean(tf.matmul(flat_vjp, flat_vjp, transpose_a=True), axis=0)
hess = tf.matmul(h, h, transpose_a=True)
print([v.shape for v in h])



M = tf.linalg.diag(1./old_p)
hl = tf.matmul(flat_vjp, M, transpose_a=True)[0]
h = tf.matmul(hl, flat_vjp)



with tf.GradientTape() as tape:
    old_policy_latent = policies[0].oldpi.policy_network(ob)
    old_pd, _ = policies[0].oldpi.pdtype.pdfromlatent(old_policy_latent)
    policy_latent = policies[0].pi.policy_network(ob)
    pd, _ = policies[0].pi.pdtype.pdfromlatent(policy_latent)
    kloldnew = old_pd.kl(pd)
    meankl = tf.reduce_mean(kloldnew)
backward = tape.gradient(meankl, policies[0].pi_var_list)


# hessian = [tf.matmul(vT, v, transpose_a=True) for vT, v in zip(flat_vjp, flat_vjp)]
# flat_tangents = []
# for v in tangents:
#     print(v.shape)
#     if len(v.shape)>1:
#         flat_tangents.append(tf.reshape(v, [int(v.shape[0])*int(v.shape[1]), 1]))
#     else:
#         flat_tangents.append(tf.reshape(v, [int(v.shape[0]), 1]))

# hvp = tf.concat([tf.squeeze(tf.matmul(h, v)) for h, v in zip(hessian, flat_tangents)], axis=0)



# hess = policies[0].my_compute_fvp(tangents, ob, ac)
# hess = policies[0].compute_fvp(tf.concat([tf.squeeze(v) for v in flat_tangents], axis=0), ob, ac, tangents)


"""