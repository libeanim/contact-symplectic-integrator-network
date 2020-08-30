import tensorflow.keras as tfk
from models.base import BaseNetwork

class ResNet(BaseNetwork):
    """ Residual Network """

    def __init__(self, step_size, horizon, name, dim_state, dim_h=500, activation='relu', **kwargs):
        super().__init__(step_size, horizon, name, dim_state)

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(dim_state)
        ])

    def step(self, x, step_size, t):
        dxdt = self.network(x)
        return x + step_size * dxdt