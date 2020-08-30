"""
CD-Lagrange Network
===================

This is the implementation of the CD-Lagrange network based on the work of
- Jean  Di  Stasio  et  al.
  “Benchmark  cases  for  robust  explicit  time  integrators  in  non-smooth  transient  dynamics”
- Fatima-Ezzahra Fekak et al.
  “A new heterogeneous asynchronous explicit–implicit timeintegrator for nonsmooth dynamics”
- Steindor Saemundsson et al.
  “Variational Integrator Networks for Physically Meaning-ful Embeddings”
"""

import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_probability as tfp
from models.base import BaseNetwork

class CDLNetwork(BaseNetwork):

  
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False, learn_friction=False, **kwargs):
        super().__init__(step_size, horizon, name, dim_state)

        self.dim_Q = self.dim_state // 2
        
        self.potential = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation, input_shape=(dim_state//2,)),
            tfk.layers.Dense(1, use_bias=False)
        ])
           
        self.contact = tfk.Sequential([
            tfk.layers.Dense(100, activation='relu', input_shape=(dim_state,)),
            tfk.layers.Dense(100, activation='relu'),
            tfk.layers.Dense(1, activation='sigmoid')
        ])
        
        self.learn_friction = learn_friction
        if self.learn_friction:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.B_param = tf.Variable(num_w*[0.], dtype=tfk.backend.floatx())
        else:
            self.B_param = None

        self.learn_inertia = learn_inertia
        if self.learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.M_param = tf.Variable(num_w*[0.], dtype=tfk.backend.floatx())
        else:
            self.M_param = None

        self.L = tf.constant([[1., -1.]])
        self.e = tf.constant([[0., 1.], [1., 0.]])


    @property
    def B(self):
        if self.learn_friction:
            B = tfp.math.fill_triangular(self.B_param)
        else:
            B = tf.linalg.diag(tf.zeros((self.dim_Q,), dtype=tfk.backend.floatx()))
        return B
    
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

    def step(self, x, step_size, t):
        u = x[:, :self.dim_Q]
        udot = x[:, self.dim_Q:]

        u_next = u + step_size * udot
        dUdu = self.grad_potential(u_next)
        damping = tf.einsum('jk,ik->ij', self.B, udot)
        w = tf.einsum('jk,ik->ij', self.M_inv,  step_size * (dUdu - damping))
        
        
        # Contact forces
        Q = tf.concat([u_next, udot], 1)
        v = -tf.einsum('jk,ik->ij', self.e, self.L * udot)
        r = v - self.L * (udot + w)
        ctf = self.contact(Q)
        r = ctf * r   
        i = tf.einsum('jk,ik->ij', self.M_inv, self.L * r)

        # Velocity next step
        udot_next = udot + w + i
        
        return tf.concat([u_next, udot_next], 1)


class CDLNetwork_Simple(CDLNetwork):
    
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False,
                 learn_friction=False, e=1., **kwargs):
        super().__init__(step_size, horizon, name, dim_state, dim_h, activation, learn_inertia, learn_friction)
        
        self.contact = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation='relu'),
            tfk.layers.Dense(dim_state//2, activation='sigmoid')
        ])
        
        self.L = tf.constant([[1.]])
        self.e = e * tf.eye(self.dim_Q)
