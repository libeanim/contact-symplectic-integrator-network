import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from models.base import BaseNetwork

class ResNet(BaseNetwork):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', pos_only=True, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only)

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(dim_state)
        ])

    def step(self, x, c, step_size, t):
        dxdt = self.network(x)
        x_next = x + step_size * dxdt
        c_next = tf.zeros_like(x_next[:, :1])

        return tf.concat([x_next, c_next], 1)


class ResNetContact(ResNet):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', pos_only=True, regularisation=0, **kwargs):
        super().__init__(step_size, horizon, name, dim_state, pos_only=pos_only, dim_h=dim_h, activation=activation, **kwargs)

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(regularisation), input_shape=(dim_state + 1,)),
            tfk.layers.Dense(dim_state, kernel_regularizer=tf.keras.regularizers.l2(regularisation))
        ])

        self.contact = tfk.Sequential([
            tfk.layers.Dense(100, activation='relu', input_shape=(dim_state,)),
            tfk.layers.Dense(100, activation='relu'),
            tfk.layers.Dense(1, activation='sigmoid')
        ])

    def step(self, x, c, step_size, t):
        
        ctf = self.contact(x)
        if c is None:
            c = np.float32(ctf.numpy() > 0.5)
            c = tf.constant(c)

        xc = tf.concat([x, c], 1)
        dxdt = self.network(xc)
        x_next = x + step_size * dxdt
        # c_next = tf.zeros_like(x_next[:, :1])

        return tf.concat([x_next, ctf], 1)


    def loss_func(self, y_true, y_pred):
        
        y_true_x = y_true[:, :, :-1]
        y_true_c = y_true[:, :, -1:]
        y_pred_x = y_pred[:, :, :-1]
        y_pred_c = y_pred[:, :, -1:]

        mse = tf.keras.losses.MSE(y_true_x, y_pred_x)
        cent = tf.keras.losses.binary_crossentropy(y_true_c, y_pred_c)
        mse = tf.reduce_mean(tf.reduce_sum(mse, 1))
        cent = tf.reduce_mean(tf.reduce_sum(cent, 1))

        return mse + cent