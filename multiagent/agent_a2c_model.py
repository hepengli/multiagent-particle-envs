from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common import colorize
from baselines.common.cg import cg
from multiagent.policies import PolicyWithValue
from multiagent.lbfgs import lbfgs
from baselines.a2c.utils import InverseLinearTimeDecay

from contextlib import contextmanager
MPI = None

class AgentA2CModel(tf.Module):
    def __init__(self, agent, network, nupdates, load_path, alpha=0.99, epsilon=1e-5, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, lr=7e-4, **network_kwargs):
        super(AgentA2CModel, self).__init__(name='AgentA2CModel')
        self.agent = agent
        self.nbs = agent.neighbors
        self.nupdates = nupdates
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

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

        with tf.name_scope(agent.name):
            if isinstance(network, str):
                network = get_network_builder(network)(**network_kwargs)
            with tf.name_scope("pi"):
                pi_policy_network = network(ob_space.shape)
                pi_value_network = network(ob_space.shape)
                self.pi = pi = PolicyWithValue(ob_space, ac_space, pi_policy_network, pi_value_network)
                lr_schedule = InverseLinearTimeDecay(initial_learning_rate=lr, nupdates=nupdates)
                self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=alpha, epsilon=epsilon)

        self.step = self.pi.step
        self.value = self.pi.value
        self.initial_state = self.pi.initial_state

        if load_path is not None:
            load_path = os.path.expanduser(load_path)
            self.ckpt = tf.train.Checkpoint(model=pi)
            load_path = os.path.join(load_path, 'agent{}'.format(self.agent.id))
            self.manager = tf.train.CheckpointManager(self.ckpt, load_path, max_to_keep=3)
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

    def convert_to_tensor(self, args):
        return [tf.convert_to_tensor(arg, dtype=arg.dtype) for arg in args if arg is not None]

    @contextmanager
    def timed(self, msg, verbose=False):
        if self.rank == 0:
            if verbose: print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            if verbose: print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    @tf.function
    def train(self, obs, rewards, actions, values):
        advs = rewards - values
        with tf.GradientTape() as tape:
            policy_latent = self.pi.policy_network(obs)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.pi.value(obs)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))
            pg_loss = tf.reduce_mean(advs * neglogpac)
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy

    def update(self, obs, states, rewards, masks, actions, values):
        args = self.convert_to_tensor((obs, states, rewards, masks, actions, values))
        if states is not None:
            states = tf.constant(states)
        policy_loss, value_loss, policy_entropy = self.train(*args)