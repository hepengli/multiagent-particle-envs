import tensorflow as tf
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from multiagent.network_ADMM import ADMM_NN
from gym import spaces

class PolicyPdClass(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation
    """

    def __init__(self, agent, observations, policy_network, sess=None, **tensors):
        """
        Parameters:
        ----------
        agent           agent i

        observations    tensorflow placeholder in which the observations will be fed

        policy_network  policy network class

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.__dict__.update(tensors)

        # Policy and value function
        self.vf = policy_network.value()
        self.logit = policy_network.policy()

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(agent.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(tf.transpose(self.logit))

        # Take an action
        action = self.pd.sample()
        if len(action.shape) == 1: action = tf.expand_dims(action, axis=1)
        self.action = tf.gather(action, agent.action_index, axis=1)
        self.neglogp = self.pd.neglogp(self.action)

        # Training op
        self.net = policy_network
        self.fit = policy_network.fit

        # Calculate the neg log of our probability
        self.sess = sess or tf.get_default_session()
        self.agent = agent

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, neglogp = self._evaluate([self.action, self.vf, self.neglogp], observation, **extra_feed)

        return a, v

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def load(self, load_path):
        name_scope = self.agent.name.replace(' ', '')
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope)
        tf_util.load_variables(load_path, variables=variables, sess=self.sess)

    def save(self, save_path):
        name_scope = self.agent.name.replace(' ', '')
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope)
        tf_util.save_variables(save_path, variables=variables, sess=self.sess)


def build_policy(env, world, agent_index, network_args, normalize_observations=True):
    env.discrete_action_input = True
    # Agent's observation parameters
    ob_dtype = env.observation_space[agent_index].dtype
    ob_shape = env.observation_space[agent_index].shape
    # Other agents' action spaces
    agent = world.agents[agent_index]
    agent = create_action_space(agent, env, world)
    # Policy network parameters
    if isinstance(agent.action_space, spaces.MultiDiscrete):
        n_outputs = int(agent.action_space.nvec.sum())
        n_friends = int(agent.action_space.nvec.size)
    else:
        n_outputs = int(agent.action_space.n)
        n_friends = 1

    def policy_fn(n_batches, observ_placeholder=None, mu_placeholder=None, sess=None):
        if observ_placeholder is None:
            X = tf.placeholder(dtype=ob_dtype, shape=(None,)+ob_shape)
        else:
            X = observ_placeholder

        extra_tensors = {}
        if normalize_observations and X.dtype == tf.float32:
            normed_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            normed_x = X

        network = ADMM_NN(
            inputs=normed_x,
            n_hiddens=network_args,
            n_outputs=n_outputs,
            n_friends=n_friends,
            n_batches=n_batches,
            dtype=X.dtype.name,
        )

        policy = PolicyPdClass(
            agent=agent,
            observations=X,
            policy_network=network,
            sess=sess,
            **extra_tensors
        )

        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

def create_action_space(agent, env, world):
    # env.discrete_action_input = True
    all_action_spaces = []
    for i, other in enumerate(world.agents):
        # store agent's action index
        if other is agent:
            agent.action_index = [len(all_action_spaces)]
        # Get parterners' action spaces
        if (hasattr(other, 'adversary') and (other.adversary==agent.adversary)) or \
        (not hasattr(other, 'adversary')):
            if agent.silent or isinstance(env.action_space[i], spaces.Discrete): # no communication action
                all_action_spaces.append(env.action_space[i])
            else:
                if isinstance(env.action_space[i], spaces.Tuple):
                    all_action_spaces.extend(env.action_space[i].spaces)
                else: # must be MultiDicrete space
                    h0, h1 = env.action_space[i].high
                    all_action_spaces.extend([spaces.Discrete(h0), spaces.Discrete(h1)])
                    if other is agent:
                        agent.action_index = agent.action_index + [agent.action_index[0]+1]

    # Augment agent's action space to include parterners' action spaces
    if len(all_action_spaces) > 1:
        # If all action spaces are discrete, simplify to MultiDiscrete action space
        if all([isinstance(ac_space, spaces.Discrete) for ac_space in all_action_spaces]):
            agent.action_space = spaces.MultiDiscrete([ac_space.n for ac_space in all_action_spaces])
        else:
            agent.action_space = spaces.Tuple(all_action_spaces)
    else:
        agent.action_space = all_action_spaces[0]
    
    return agent