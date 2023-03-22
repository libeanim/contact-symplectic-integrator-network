import numpy as np
import tensorflow as tf

class Environment:
    """
    Environment Base Class
    ----------------------

    Environment objects contain all the information required to simulate a system.
    This includes the methods to generate data as well as all paramteres needed in the system.
    
    """

    def __init__(self, DATA, CONTACT, steps=500, dt=0.01, horizon=10, SIGMA=0., SEED=0, epochs=3000):
        self.dt = dt
        self.steps = steps
        self.horizon = horizon
        self.SIGMA = SIGMA
        self.SEED = SEED
        self.epochs = epochs
        self.DATA = DATA
        self.CONTACT = CONTACT
        self.trajectory = None
        tf.random.set_seed(SEED)

    def get_filename(self, name, suffix='png', folder='images'):
        if folder is None:
            folder = '.'
        return '{}/{}-{}-{}-{}-{}.{}'.format(folder, self.SEED, self.DATA, self.CONTACT, self.epochs, name, suffix)

    def generate(self):
        raise NotImplementedError('Please implement this method.')

    def plot(self):
        raise NotImplementedError('Please implement this method.')
    
    def prepare_output(self):
        """Add noise to the trajectory"""

        X, y, c = [], [], []
        np.random.seed(self.SEED)
        gaussian_noise = np.random.normal(0, self.SIGMA, [self.trajectory.shape[0], self.trajectory.shape[1] - 1])
        
        noisy_trajectory = np.hstack([
            self.trajectory[:, :-1] + gaussian_noise,  # Only add noise to position
            self.trajectory[:, -1:]                    # (implicitly means touch is noisy too)
        ])
        
        for i in range(noisy_trajectory.shape[0] - self.horizon - 1):
            X.append(noisy_trajectory[i:i + 1, :-1].flatten())
            y.append(noisy_trajectory[i + 1:i + self.horizon + 1, :-1].flatten())
            c.append(noisy_trajectory[i + 1:i + self.horizon + 1, -1:].flatten())
        self.X, self.y, self.c = np.array(X), np.array(y), np.array(c)
        return self.X, self.y, self.c
