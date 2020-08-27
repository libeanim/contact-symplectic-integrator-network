import tensorflow.keras as tfk
import tensorflow as tf

class BaseNetwork(tfk.Model):

    def __init__(self, step_size, horizon, name, **kwargs):

        super().__init__(name=name)

        self.step_size = step_size
        self.horizon = horizon

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._build_network(**kwargs)

        inputs = tfk.Input(shape=(None, self.dim_state), dtype=tfk.backend.floatx())
        outputs = self.call(inputs)

    def call(self, x0):
        return self.forward(x0, self.step_size, self.horizon)[:, 1:]

    def forward(self, x0, step_size, horizon):
        x = [x0]
        for t in range(horizon):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)