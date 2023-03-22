import numpy as np
import tensorflow as tf

def TRAIN(env, model, name, learning_rate=1e-2, loss='mse', pos_only=False, initialise=True,
          learn_friction=False, verbose=2):
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
        m = model(env.dt, env.horizon, name=name, dim_state=env.X.shape[1], e=e, pos_only=pos_only,
                  learn_inertia=False, learn_friction=learn_friction, activation=tf.nn.softplus)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        m.compile(optimizer, loss=m.loss_func, run_eagerly=False)
    else:
        m = model
    loss = []
    log = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, log: loss.append([epoch, log['loss']]))

    X = tf.convert_to_tensor(env.X.reshape(env.X.shape[0], 1, env.X.shape[1]), np.float32)
    c = tf.convert_to_tensor(env.c.reshape(env.c.shape[0], env.horizon, 1), np.float32)

    if pos_only:
        y = tf.convert_to_tensor(env.y.reshape(env.y.shape[0], env.horizon,
                                               env.X.shape[1])[:, :, :env.X.shape[1]//2], np.float32)
    else:
        y = tf.convert_to_tensor(env.y.reshape(env.y.shape[0], env.horizon, env.X.shape[1]), np.float32)
    y = tf.concat([y, c], 2)
    m.fit([X, c], y, epochs=env.epochs, shuffle=True, callbacks=[log], verbose=verbose)

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
    return model.predict_forward(
        tf.convert_to_tensor([[env.trajectory[0, :-1]]], np.float32),
        env.dt,
        env.trajectory.shape[0]
    )[0]