from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ADMM_NN(object):
    """ Class for ADMM Neural Network. """

    def __init__(self, inputs, n_hiddens, n_outputs, n_friends, n_batches=None, dtype='float32'):
        """
        Initialize variables for NN.
        Raises:
            ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        :param n_inputs: Number of inputs
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_friends: Number of friends
        :param return:
        """
        self.s = tf.transpose(inputs)
        self.n_inputs = n_inputs = int(inputs.shape[1])
        self.n_hiddens = n_hiddens = [n_hiddens] if not isinstance(n_hiddens, list) else n_hiddens
        self.n_nodes = [n_inputs] + n_hiddens
        self.n_friends = n_friends
        self.n_outputs = n_outputs
        self.n_batches = n_batches
        self._dtype = dtype

        def make_variables(name_scope, shape, initializer=None): 
            if initializer is not None:
                return tf.get_variable(name_scope, initializer=initializer(shape, dtype))
            else:
                return tf.get_variable(name_scope, shape, dtype)

        n_nodes = self.n_nodes + [n_outputs]
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            # NN weights:
            self.w = [make_variables('w{}'.format(i), [n_nodes[i+1], n_nodes[i]], tf.keras.initializers.Orthogonal()) for i in range(len(n_nodes)-1)]
            # Neurons' inputs and outputs
            self.x = [make_variables('x{}'.format(i), [n_node, n_batches], tf.random_uniform_initializer()) for i, n_node in enumerate(n_hiddens)]
            self.o = [make_variables('o{}'.format(i), [n_node, n_batches], tf.random_uniform_initializer()) for i, n_node in enumerate(n_hiddens)]
            # NN' outputs
            logit = make_variables('logit', [n_outputs, n_batches], tf.random_uniform_initializer())
            self.lam = make_variables('lam', [n_outputs, n_batches], tf.ones_initializer())
            self.x.append(logit)
            # Estimate of neighbours' outpus
            self.z = make_variables('z', [n_friends, n_outputs, n_batches], tf.zeros_initializer())
            self.p = make_variables('p', [n_friends, n_outputs, n_batches], tf.zeros_initializer())
        n_nodes = self.n_nodes + [1]
        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            # Value NN weights
            self.v = [make_variables('v{}'.format(i), [n_nodes[i+1], n_nodes[i]]) for i in range(len(n_nodes)-1)]

    def pinv(self, a, rcond=1e-15):
        s, u, v = tf.svd(a)
        # Ignore singular values close to zero to prevent numerical overflow
        limit = rcond * tf.reduce_max(s)
        non_zero = tf.greater(s, limit)

        reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros_like(s))
        lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
        return tf.matmul(lhs, u, transpose_b=True)

    def _relu(self, x):
        return tf.maximum(0.,x)

    def _weight_update(self, x, o, beta):
        """
        min_{W_l} beta * ||x_l - W_l * o_{l-1}||^2

        W_l = (beta * x_l * O_{l-1}^T) * inv(beta * o_{l-1} * O_{l-1}^T)

        OR

        W_l = x_l * pinv

        return: W_l
        """
        return tf.matmul(x, self.pinv(o))

    def _activation_update(self, x_next, W_next, x, beta, alpha):
        """
        min_{o_l} beta * ||x_{l+1} - W_{l+1} * o_l||^2 + alpha * ||o_l - h(x_l)||^2

        o_l = inv(W_{l+1}^T * W_{l+1} + alpha * I) * (beta * W_{l+1}^T * x_{l+1} + alpha * h(x_l))

        return: o_l
        """
        # Activation inverse
        m1 = tf.matmul(W_next, W_next, transpose_a=True)
        m2 = tf.scalar_mul(alpha, tf.eye(int(m1.shape[0]), dtype=self._dtype))
        av = tf.linalg.inv(m1+m2)

        # Activation formulate
        m3 = beta * tf.matmul(W_next, x_next, transpose_a=True)
        m4 = alpha * self._relu(x)
        af = m3 + m4

        # Output
        return tf.matmul(av, af)

    def _argmin_x(self, o, w, o_last, beta, alpha):
        """
        min_{x_l} alpha * ||o_l - h(x_l)||^2 + beta * ||x_l - W_l * O_{l-1}||^2
        
        x_l = (alpha * o_l + beta * W_l * O_{l-1}) / (alpha + beta),  if x_l > 0

            =  W_l * O_{l-1}, otherwise
        """
        m = tf.matmul(w, o_last)
        sol1 = (alpha * o + beta * m) / (alpha + beta)
        sol2 = m
        x1 = tf.zeros_like(o)
        x2 = tf.zeros_like(o)

        x1 = tf.where(tf.less_equal(x1, sol1), sol1, x1)
        x2 = tf.where(tf.less_equal(sol2, x2), sol2, x2)

        f_1 = alpha * tf.square(o - self._relu(x1)) + beta * (tf.square(x1 - m))
        f_2 = alpha * tf.square(o - self._relu(x2)) + beta * (tf.square(x2 - m))

        return tf.where(tf.less_equal(f_1, f_2), x1, x2)

    def _argmin_logit(self, target, w, o, z, p, lam, beta, rho, eta, comm):
        """
        min_{logit} eta * ||logit - target||^2 +
                    lam * (logit - w * o) + beta * ||logit - w * o||^2 + 
                    p .* (comm * logit - z) + rho * ||comm * logit - z||^2

        # A should be 1 or -1

        return: logit
        """
        # ha_k = tf.reduce_sum(tf.tensordot(h, a_k, [[1], [0]]), axis=1)
        m = eta * target - lam + beta * tf.matmul(w, o)# - p * comm + rho * comm * z
        v = eta + beta# + rho * comm * comm

        # No friends
        if self.n_friends == 1:
            m = eta * target - lam + beta * tf.matmul(w, o)
            v = eta + beta

        return m / v

    def _lam_update(self, logit, w, o, beta):
        return beta*(logit - tf.matmul(w, o))

    def _z_and_p_update(self, a, a_neighbor, p, p_neighbor, A, A_neighbor, rho):
        v = 0.5 * (p + p_neighbor) + 0.5 * rho * (A * a + A_neighbor * a_neighbor)
        z = (1.0/rho) * (p - v) + A * a
        p = v
        return z, p

    def policy(self):
        mu = self.s
        for i, w in enumerate(self.w):
            mu = tf.matmul(w, mu)
            mu = self._relu(mu) if i<(len(self.w)-1) else mu

        return mu

    def value(self):
        vf = self.s
        for i, v in enumerate(self.v):
            vf = tf.matmul(v, vf)
            vf = self._relu(vf) if i<(len(self.v)-1) else vf

        return tf.transpose(vf)

    def fit(self, target, comm, neighbor_id, alpha, rho, beta, eta):
        w_new, o_new, x_new = [], [], []
        for n in range(len(self.n_nodes)):
            if n == 0:
                # Input layer
                w = self._weight_update(self.x[n], self.s, beta)
                o = self._activation_update(self.x[n+1], self.w[n+1], self.x[n], beta, alpha)
                x = self._argmin_x(o, w, self.s, beta, alpha)
                w_new.append(w)
                o_new.append(o)
                x_new.append(x)
            elif n < len(self.n_nodes) - 1:
                # Hidden layer
                w = self._weight_update(self.x[n], o, beta)
                o = self._activation_update(self.x[n+1], self.w[n+1], self.x[n], beta, alpha)
                x = self._argmin_x(o, w, o_new[-1], beta, alpha)
                w_new.append(w)
                o_new.append(o)
                x_new.append(x)
            else:
                w = self._weight_update(self.x[-1], o, beta)
                z = tf.gather(self.z, 0)
                p = tf.gather(self.p, 0)
                lam = self.lam
                logit = self._argmin_logit(target, w, o, z, p, lam, beta, rho, eta, comm)
                lam_new = self._lam_update(logit, w, o, beta)
                w_new.append(w)
                x_new.append(logit)

        update_op = [tf.assign(self.w[i], w_new[i]) for i in range(len(self.w))] + \
                    [tf.assign(self.o[i], o_new[i]) for i in range(len(self.o))] + \
                    [tf.assign(self.x[i], x_new[i]) for i in range(len(self.x))] + \
                    [tf.assign(self.lam, lam_new)]

        return update_op

    def info_to_exchange(self, neighbor_id):
        a = self.policy()
        p = tf.gather(self.p, neighbor_id)

        return a, p

    def exchange(self, sess, OB, neighbor_id, rho):
        a, p = self.info_to_exchange(neighbor_id)
        def exchange_fn(neighbor, a_neighbor, p_neighbor, A_neighbor, obs, A):
            z_new, p_new = self._z_and_p_update(a, a_neighbor, p, p_neighbor, A, A_neighbor, rho)
            update_op = [tf.assign(self.z[neighbor], z_new), tf.assign(self.p[neighbor], p_new)]
            sess.run(update_op, feed_dict={OB: obs, neighbor_id:neighbor})

        return exchange_fn

    def evaluate(self, inputs, labels):
        """
        Calculate loss
        :param inputs: inputs data
        :param outputs: ground truth
        :return: loss
        """
        loss = tf.reduce_mean(tf.square(self.value() - labels))

        return loss

    # def warming(self, inputs, labels, epochs, beta, alpha, rho, A):
    #     """
    #     Warming ADMM Neural Network by minimizing sub-problems without update lambda
    #     :param inputs: input of training data samples
    #     :param outputs: label of training data samples
    #     :param epochs: number of epochs
    #     :param beta: value of beta
    #     :param alpha: value of alpha
    #     :return:
    #     """
    #     self.a0 = inputs
    #     for i in range(epochs):
    #         print("------ Warming: {:d} ------".format(i))

    #         for m in range(self.x1.shape[0]):
    #         # m = np.random.choice(A.shape[0])
    #             q = np.where(A[m] != 0)[0]
    #             # a)
    #             for n in q:
    #                 # Input layer
    #                 self.w1[n] = self._weight_update(self.z1[n], self.a0[n], beta, rho, self.y1[m,n], self.x1[m,n], A[m,n])
    #                 self.a1[n] = self._activation_update(self.w2[n], self.z2[n], self.z1[n], beta, alpha)
    #                 self.z1[n] = self._argminz(self.a1[n], self.w1[n], self.a0[n], beta, alpha)
    #                 # Hidden layer
    #                 self.w2[n] = self._weight_update(self.z2[n], self.a1[n], beta, rho, self.y2[m,n], self.x2[m,n], A[m,n])
    #                 self.a2[n] = self._activation_update(self.w3[n], self.z3[n], self.z2[n], beta, alpha)
    #                 self.z2[n] = self._argminz(self.a2[n], self.w2[n], self.a1[n], beta, alpha)
    #                 # Output layer
    #                 self.w3[n] = self._weight_update(self.z3[n], self.a2[n], beta, rho, self.y3[m,n], self.x3[m,n], A[m,n])
    #                 self.z3[n] = self._argminlastz(labels[n], self.lambda_larange[n], self.w3[n], self.a2[n], beta)
    #                 self.lambda_larange[n] = self._lambda_update(self.z3[n], self.w3[n], self.a2[n], beta)
    #             # b)
    #             v1 = 0.5 * (self.y1[m,q[0]] + self.y1[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w1[q[0]] + A[m,q[1]]*self.w1[q[1]])
    #             v2 = 0.5 * (self.y2[m,q[0]] + self.y2[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w2[q[0]] + A[m,q[1]]*self.w2[q[1]])
    #             v3 = 0.5 * (self.y3[m,q[0]] + self.y3[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w3[q[0]] + A[m,q[1]]*self.w3[q[1]])
    #             for n in q:
    #                 self.x1[m,n] = (1.0/rho) * (self.y1[m,n] - v1) + A[m,n] * self.w1[n]
    #                 self.x2[m,n] = (1.0/rho) * (self.y2[m,n] - v2) + A[m,n] * self.w2[n]
    #                 self.x3[m,n] = (1.0/rho) * (self.y3[m,n] - v3) + A[m,n] * self.w3[n]
    #             # c)
    #                 self.y1[m,n] = v1
    #                 self.y2[m,n] = v2
    #                 self.y3[m,n] = v3


    # def drawcurve(self, train_, valid_, id, legend_1, legend_2):
    #     acc_train = np.array(train_).flatten()
    #     acc_test = np.array(valid_).flatten()

    #     plt.figure(id)
    #     plt.plot(acc_train)
    #     plt.plot(acc_test)

    #     plt.legend([legend_1, legend_2], loc='upper left')
    #     plt.draw()
    #     plt.pause(0.001)
    #     return 0
