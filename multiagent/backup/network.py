import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import Model


class PolicyNetwork(Model):
    def __init__(self, input_shape, n_hidden, n_output, n_friend, n_batch=None, dtype=tf.float32):
        super(PolicyNetwork, self).__init__()
        assert isinstance(n_hidden, list)
        self.bn = BatchNormalization(input_shape=input_shape, dtype=dtype)
        self.d = []
        for n_nodes in n_hidden:
            self.d.append(Dense(n_nodes, activation='relu', dtype=dtype))
        self.o = Dense(n_output, dtype=dtype)
        # ADMM paras
        self.z = tf.ones([n_friend, n_batch], dtype=dtype)
        self.p = tf.zeros([n_friend, n_batch], dtype=dtype)

    def call(self, x, training):
        x = self.bn(x)
        for d in self.d: x = d(x)
        return self.o(x)





# class ADMM_NN(object):
#     """ Class for ADMM Neural Network. """

#     def __init__(self, inputs, n_hiddens, n_outputs, n_friends, n_batches=None, dtype='float32'):
#         """
#         Initialize variables for NN.
#         Raises:
#             ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
#         :param n_inputs: Number of inputs
#         :param n_hiddens: Number of hidden units.
#         :param n_outputs: Number of outputs
#         :param n_friends: Number of friends
#         :param return:
#         """
#         self.s = inputs
#         self.n_inputs = n_inputs = int(inputs.shape[1])
#         self.n_hiddens = n_hiddens = [n_hiddens] if not isinstance(n_hiddens, list) else n_hiddens
#         self.n_nodes = [n_inputs] + n_hiddens
#         self.n_friends = n_friends
#         self.n_outputs = n_outputs
#         self.n_batches = n_batches
#         self._dtype = dtype

#         def make_variables(name_scope, shape, initializer=None): 
#             if initializer is not None:
#                 return tf.get_variable(name_scope, initializer=initializer(shape, dtype))
#             else:
#                 return tf.get_variable(name_scope, shape, dtype)

#         # Policy network
#         n_nodes = self.n_nodes + [n_outputs]
#         with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
#             # NN weights:
#             self.w = [make_variables('w{}'.format(i), [n_nodes[i+1], n_nodes[i]], tf.keras.initializers.Orthogonal()) for i in range(len(n_nodes)-1)]
#             self.b = [make_variables('b{}'.format(i), [n_nodes[i+1]], tf.ones_initializer()) for i in range(len(n_nodes)-1)]
#             # Estimate of neighbours' outpus
#             self.z = make_variables('z', [n_friends, n_batches], tf.ones_initializer())
#             self.p = make_variables('p', [n_friends, n_batches], tf.zeros_initializer())

#         # Value network
#         n_nodes = self.n_nodes + [1]
#         with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
#             # Value NN weights
#             self.v = [make_variables('v{}'.format(i), [n_nodes[i+1], n_nodes[i]]) for i in range(len(n_nodes)-1)]

#     @property
#     def flat_var_list(self):
#         var_list = self.w + self.b

#         return tf.concat(axis=0, values=[tf.reshape(v, [self.numel(v)]) for v in var_list])

#     def var_shape(self, x):
#         out = x.get_shape().as_list()
#         assert all(isinstance(a, int) for a in out), \
#             "shape function assumes that shape is fully known"
#         return out

#     def numel(self, x):
#         return self.intprod(self.var_shape(x))

#     def intprod(self, x):
#         return int(np.prod(x))

#     def pinv(self, a, rcond=1e-15):
#         s, u, v = tf.svd(a)
#         # Ignore singular values close to zero to prevent numerical overflow
#         limit = rcond * tf.reduce_max(s)
#         non_zero = tf.greater(s, limit)

#         reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros_like(s))
#         lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
#         return tf.matmul(lhs, u, transpose_b=True)

#     def _relu(self, x):
#         return tf.maximum(0.,x)

#     def _z_and_p_update(self, a, a_neighbor, p, p_neighbor, A, A_neighbor, rho):
#         v = 0.5 * (p + p_neighbor) + 0.5 * rho * (A * a + A_neighbor * a_neighbor)
#         z = (1.0/rho) * (p - v) + A * a
#         p = v
#         return z, p

#     def policy(self):
#         logit = self.s
#         for i, (w, b) in enumerate(zip(self.w, self.b)):
#             logit = tf.matmul(logit, w, transpose_b=True) + b
#             logit = self._relu(logit) if i<(len(self.w)-1) else logit

#         return logit

#     def value(self):
#         vf = self.s
#         for i, v in enumerate(self.v):
#            vf = tf.matmul(vf, v, transpose_b=True)
#            vf = self._relu(vf) if i<(len(self.v)-1) else vf

#         return vf

#     def info_to_exchange(self, neighbor_id):
#         a = self.policy() # self.x[-1]
#         p = tf.gather(self.p, neighbor_id)#[0,:]

#         return p

#     def exchange(self, sess, OB, AC, CLIPRANGE, neighbor_id, rho):
#         # a, p = self.info_to_exchange(neighbor_id)
#         def exchange_fn(neighbor, a, a_neighbor, p, p_neighbor, A_neighbor, obs, actions, cliprange, A):
#             z_new, p_new = self._z_and_p_update(a, a_neighbor, p, p_neighbor, A, A_neighbor, rho)
#             update_op = [tf.assign(self.z[neighbor], z_new), tf.assign(self.p[neighbor], p_new)]
#             sess.run(update_op, feed_dict={OB: obs, AC: actions, CLIPRANGE: cliprange, neighbor_id:neighbor})

#         return exchange_fn
