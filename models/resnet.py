import tensorflow as tf
import tensorflow.keras as tfk
from models.base import BaseNetwork

class ResNet(BaseNetwork):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only=pos_only)

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(dim_state)
        ])

    def step(self, x, c, step_size, t):
        
        #xc = tf.concat([x, c], 1)
        dxdt = self.network(x)
        x_next = x + step_size * dxdt
        c_next = tf.zeros_like(x_next[:, :1])

        return tf.concat([x_next, c_next], 1)