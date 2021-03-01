"""
VIN Model
=========

This model is based on the article
"Variational Integrator Networks for Physically Structured Embeddings"
by Steindor Saemundsson, Alexander Terenin, Katja Hofmann, Marc Peter Deisenroth
https://arxiv.org/abs/1910.09349

Code provided by Steindor Saemundsson
Modified by Andreas Hochlehnert to work in this context

Date: August 2020
"""

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from models.base import BaseNetwork

class VIN(BaseNetwork):
    """ Variational Integrator Network """

    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False, pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        self.dim_Q = self.dim_state // 2

        self.potential = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(1, use_bias=False)
        ])

        self.learn_inertia = learn_inertia
        if self.learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.L_param = tf.Variable(num_w*[0.], dtype=tfk.backend.floatx())
        else:
            self.L_param = None

    @property
    def M_inv(self):
        if self.learn_inertia:
            L = tfp.math.fill_triangular(self.L_param)
            M_inv = tf.transpose(L) @ L
        else:
            M_inv = tf.linalg.diag(tf.ones((self.dim_Q,), dtype=tfk.backend.floatx()))
        return M_inv

    def grad_potential(self, q):

        with tf.GradientTape() as g:
            g.watch(q)
            U = self.potential(q)

        return g.gradient(U, q)

        
    def loss_func(self, y_true, y_pred):
        y_true_x = y_true[:, :, :-1]
        y_pred_x = y_pred[:, :, :-1]

        mse = tf.keras.losses.MSE(y_true_x, y_pred_x)
        mse = tf.reduce_mean(tf.reduce_sum(mse, 1))
        return mse

    def step(self, x, c, step_size, t):
        raise NotImplementedError()


class VIN_SV(VIN):
    """ Störmer-Verlet VIN """

    def step(self, x, c, step_size, t):

        c_next = tf.zeros_like(x[:, :1])
        q = x[:, :self.dim_Q]
        q_prev = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)

        qddot = tf.einsum('jk,ik->ij', self.M_inv, dUdq)
        q_next = 2 * q - q_prev - (step_size**2) * qddot

        return tf.concat([q_next, q, c_next], 1)

    def forward(self, q0, step_size, horizon):

        x0 = tf.concat([q0[:, :1], tf.zeros_like(q0[:, :1])], 2)
        x1 = tf.concat([q0[:, 1:2], q0[:, :1]], 2)
        x0 = tf.concat([x0, x1], 1)
        x = [x0]
        for t in range(horizon-1):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)


class VIN_VV(VIN):
    """ Velocity-Verlet VIN """

    def step(self, x, c, step_size, t):

        c_next = tf.zeros_like(x[:, :1])
        q = x[:, :self.dim_Q]
        qdot = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)
        
        qddot = tf.einsum('jk,ik->ij', self.M_inv, dUdq)

        q_next = q + step_size * qdot - 0.5 * (step_size**2) * qddot
        dUdq_next = self.grad_potential(q_next)

        dUdq_mid = dUdq + dUdq_next
        qddot_mid = tf.einsum('jk,ik->ij', self.M_inv, dUdq_mid)

        qdot_next = qdot - 0.5 * step_size * qddot_mid

        return tf.concat([q_next, qdot_next, c_next], 1)