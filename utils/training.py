import numpy as np
import tensorflow as tf

def TRAIN(env, model, name, learning_rate=1e-3, loss='mse', initialise=True, learn_friction=False):
    """
    Train a model given an environment and the learning parameters.

    Parameters
    ----------
        env:            Environment
        model:          Model class or model
        learning_rate:  Learning rate
        loss:           Loss type
        initialise:     Initialise the model.
                        If False model will not be reinitialised so that training can continue.
    """
    if initialise:
        e = 1.0
        if hasattr(env, 'e'):
            e = env.e
        m = model(env.dt, env.horizon, name=name, dim_state=env.X.shape[1], e=e,
                  learn_inertia=False, learn_friction=learn_friction, activation='tanh')
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        m.compile(optimizer, loss=loss, run_eagerly=False)
    else:
        m = model
    loss = []
    log = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, log: loss.append([epoch, log['loss']]))
    m.fit(
        tf.convert_to_tensor(env.X.reshape(env.X.shape[0], 1, env.X.shape[1]), np.float32),
        tf.convert_to_tensor(env.y.reshape(env.y.shape[0], env.horizon, env.X.shape[1]), np.float32),
        epochs=env.epochs, shuffle=True, callbacks=[log])
    m.loss_data = np.array(loss)
    return m

def PREDICT(env, model):
    """
    Predict the trajectory given the environment and the trained model

    Parameters
    ----------
        env:    Environment
        model:  Trained model
    """
    return model.forward(tf.convert_to_tensor([[env.trajectory[0]]], np.float32), env.dt, env.trajectory.shape[0])[0]