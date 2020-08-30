import tensorflow as tf

class BaseNetwork(tf.keras.Model):

    def __init__(self, step_size, horizon, name, dim_state):

        super().__init__(name=name)

        self.step_size = step_size
        self.horizon = horizon
        self.dim_state = dim_state


    def call(self, x0):
        return self.forward(x0, self.step_size, self.horizon)[:, 1:]

    def forward(self, x0, step_size, horizon):
        x = [x0]
        for t in range(horizon):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)