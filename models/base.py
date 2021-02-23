import tensorflow as tf

class BaseNetwork(tf.keras.Model):

    def __init__(self, step_size, horizon, name, dim_state, pos_only):

        super().__init__(name=name)

        self.step_size = step_size
        self.horizon = horizon
        self.dim_state = dim_state
        self.pos_only = pos_only


    def call(self, xc):
        x0, c = xc
        if self.pos_only:
            xc_out = self.forward(x0, c, self.step_size, self.horizon)
            xpos = xc_out[:, 1:, :self.dim_state // 2]
            return tf.concat([xpos, xc_out[:, 1:, -1:]], 2)
        else:
            return self.forward(x0, c, self.step_size, self.horizon)[:, 1:]

    def forward(self, x0, c_true, step_size, horizon):
        x = [x0]
        c = [tf.zeros_like(c_true[:, :1])]
        for t in range(horizon):
            x_t = x[-1][:, -1]
            c_t = c_true[:, t]
            xc_next = self.step(x_t, c_t, step_size, t)
            x_next = xc_next[:, :-1]
            c_pred = xc_next[:, -1:]
            x.append(x_next[:, None])
            c.append(c_pred[:, None])

        return tf.concat([tf.concat(x, 1), tf.concat(c, 1)], 2)

    def predict_forward(self, x0, step_size, horizon):
        x = [x0]
        c = []
        for t in range(horizon):
            x_t = x[-1][:, -1]
            xc_next = self.step(x_t, None, step_size, t)
            x_next = xc_next[:, :-1]
            c_pred = xc_next[:, -1:]
            x.append(x_next[:, None])
            c.append(c_pred[:, None])

        c = tf.concat(c, 1)
        c = tf.concat([tf.zeros_like(c[:, :1]), c], 1)
        return tf.concat([tf.concat(x, 1), c], 2)

    def loss_func(self, y_true, y_pred):

        y_true_x = y_true[:, :, :-1]
        y_pred_x = y_pred[:, :, :-1]

        mse = tf.keras.losses.MSE(y_true_x, y_pred_x)
        mse = tf.reduce_mean(tf.reduce_sum(mse, 1))
        return mse