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

import numpy as np
import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_probability as tfp
from models.base import BaseNetwork

class CDLNetwork(BaseNetwork):
    """
    CD-Lagrange Network
    ===================

    step_size:      `float`;
                    The time step size "h" used for integration.
    horizon:        `int`;
                    The number of forward steps the network has to predict during training.
    name:           `str`;
                    Name of the network
    dim_state:      `int`;
                    Number of states the system has.
    dim_h:          `int`;
                    Number of units in the hidden layer.
    activation:     `str`;
                    Activation function used in the potential network.
    learn_inertia:  `bool`;
                    Determines whether to learn the mass matrix
    learn_friction: `bool`;
                    Determines whether to learn the friction paramter.
    """

  
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='tanh',
                learn_inertia=False, learn_friction=False, pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        self.dim_Q = self.dim_state // 2
        
        self.potential = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation, input_shape=(dim_state//2,),
                             kernel_regularizer=tf.keras.regularizers.l2(0.05)),
            tfk.layers.Dense(1, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.05))
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
        """Friction paramter"""
        if self.learn_friction:
            B = tfp.math.fill_triangular(self.B_param)
        else:
            B = tf.linalg.diag(tf.zeros((self.dim_Q,), dtype=tfk.backend.floatx()))
        return B
    
    @property
    def M_inv(self):
        """Inverse of mass matrix"""
        if self.learn_inertia:
            L = tfp.math.fill_triangular(self.L_param)
            M_inv = tf.transpose(L) @ L
        else:
            M_inv = tf.linalg.diag(tf.ones((self.dim_Q,), dtype=tfk.backend.floatx()))
        return M_inv

    def grad_potential(self, q):
        """Gradient of the potential"""
        with tf.GradientTape() as g:
            g.watch(q)
            U = self.potential(q)

        return g.gradient(U, q)

    def step(self, x, c, step_size, t):
        """Calculate next step using the CD-Lagrange integrator.""" 
        u = x[:, :self.dim_Q] 
        udot = x[:, self.dim_Q:] 

        u_next = u + step_size * udot 
        dUdu = self.grad_potential(u_next) 
        damping = tf.einsum('jk,ik->ij', self.B, udot) 
        w = tf.einsum('jk,ik->ij', self.M_inv,  step_size * (dUdu - damping)) 
        
        
        # Contact forces 
        Q = tf.concat([u_next, udot], 1)  

        v = tf.einsum('jk,ik->ij', self.e, self.L * udot) 
        r = v - self.L * (udot + w) 
        ctf = self.contact(Q) 

        if c is None: 
            c = np.float32(ctf.numpy() > 0.5) 
            c = tf.constant(c) 


        #closest point projection
        u_next = tf.where(tf.greater(ctf, tf.constant([0.5])), 0., u_next)
        r = c * r

        i = tf.einsum('jk,ik->ij', self.M_inv, self.L * r) 

        # Velocity next step 
        udot_next = udot + w + i 
        
        return tf.concat([u_next, udot_next, ctf], 1)

    def loss_func(self, y_true, y_pred):

        y_true_x = y_true[:, :, :-1]
        y_true_c = y_true[:, :, -1:]
        y_pred_x = y_pred[:, :, :-1]
        y_pred_c = y_pred[:, :, -1:]

        mse = tf.keras.losses.MSE(y_true_x, y_pred_x)
        cent = tf.keras.losses.binary_crossentropy(y_true_c, y_pred_c)
        mse = tf.reduce_mean(tf.reduce_sum(mse, 1))
        cent = tf.reduce_mean(tf.reduce_sum(cent, 1))

        return cent + mse


class CDLNetwork_Simple(CDLNetwork):
    
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False,
                 learn_friction=False, e=1., pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, dim_h, activation, learn_inertia, learn_friction, pos_only)
        
        # self.contact = tfk.Sequential([
        #     tfk.layers.Dense(dim_h, activation='relu'),
        #     tfk.layers.Dense(dim_state//2, activation='sigmoid')
        # ])
        
        self.L = tf.constant([[1.]])
        self.e = e * tf.eye(self.dim_Q)

class CDLNetwork_NoContact(CDLNetwork_Simple):
    
    def __init__(self, step_size, horizon, name, dim_state=10, dim_h=500, activation='relu', learn_inertia=False,
                 learn_friction=False, e=1., **kwargs):
        super().__init__(step_size, horizon, name, dim_state, dim_h, activation, learn_inertia,
                         learn_friction, e)
        
        self.contact = lambda a: tf.zeros_like(a)

        
