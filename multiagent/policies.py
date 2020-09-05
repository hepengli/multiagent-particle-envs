import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import gym


class PolicyWithValue(tf.keras.Model):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ob_space, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """
        super(PolicyWithValue, self).__init__()

        self.policy_network_fn = policy_network
        self.value_network_fn = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None
        self.ob_rms = RunningMeanStd(shape=ob_space.shape, default_clip_range=5.0)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(policy_network.output_shape, ac_space, init_scale=0.01)

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network_fn.output_shape, 'q', ac_space.n)
        else:
            self.value_fc = fc(self.value_network_fn.output_shape, 'vf', 1)

    def policy_network(self, observation):
        return self.policy_network_fn(self.ob_rms.normalize(observation))

    def value_network(self, observation):
        return self.value_network_fn(self.ob_rms.normalize(observation))

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        latent = self.policy_network(observation)
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        neglogp = pd.neglogp(action)
        value_latent = self.value_network(observation)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """
        value_latent = self.value_network(observation)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

