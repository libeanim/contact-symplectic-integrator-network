import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from environments.base import Environment

class Pendulum(Environment):
    """
    Pendulum Environment

    Parameters:
    -----------

        steps:      int;
                    number of time steps
        dt:         float;
                    time steps size
        epochs:     int;
                    number of epochs
        CONTACT:    str;
                    contact type
        mass:       float;
                    mass of the ball
        g:          float;
                    accelaration constant
        friction:   float;
                    damping parameter
        length:     float;
                    length of pendulum thread
        horizon:    int;
                    prediction time window
        SIGMA:      float;
                    variance of gaussian white noise
        SEED:       int;
                    random seed
    """


    MODEL=('runge-kutta', 'analytical-small-angle')

    def __init__(
            self, steps=500, dt=0.02, epochs=3000, CONTACT='nomax', mass=1., g=-9.81,
            friction=0., length=1., horizon=10, SIGMA=0., SEED=0):
        DATA = 'pendulum' if friction == 0. else 'pendulum_friction'
        super().__init__(DATA, CONTACT, steps, dt, horizon, SIGMA, SEED, epochs)
        self.friction = friction
        self.mass = mass
        self.g = g
        self.length = length

    def generate(self, model=MODEL[0], state0=[1, 0]):
        if model == self.MODEL[0]:
            return self.pendulum(state0)
        elif model == self.MODEL[1]:
            return self.pendulum_friction_small_angle(state0)
        else:
            raise ValueError(
                'Please choose an available model ({})'.format(', '.join(self.MODEL))
            )

    def pendulum(self, y0):

        T = self.steps * self.dt
        t = np.linspace(0, T, self.steps)

        def pendulum_ODE(t, y):
            q, qdot = y
            dydt = [qdot, -self.friction*qdot + (self.g/self.length)*np.sin(q)]
            return dydt
        
        def integrate_ODE(num_steps, step_size, y0, rtol=1e-12):

            T = num_steps * step_size
            t = np.linspace(0.0, T, num_steps)

            solver = ode(pendulum_ODE).set_integrator('dop853', rtol=rtol)
            sol = np.empty((len(t), 2))
            sol[0] = y0
            solver.set_initial_value(y0)
            k = 1
            while solver.successful() and solver.t < T:
                solver.integrate(t[k])
                sol[k] = solver.y
                k += 1
            
            return sol

        def run(num_traj, num_steps, step_size, y0):

            qqd = []
            for n in range(num_traj):
                y0_n = y0[n]
                qqd_n = integrate_ODE(num_steps, step_size, y0=y0_n)
                qqd.append(qqd_n[None])
            qqd = np.vstack(qqd)

            return qqd

        data = run(1, self.steps, self.dt, (y0,))
        self.trajectory = data[0]
        self.prepare_output()
        
        return self.X, self.y, (t, self.trajectory)

    def pendulum_friction_small_angle(self, state0):
        q0, qd0 = state0[0], state0[1]
        T = self.steps * self.dt
        t = np.linspace(0, T, self.steps)

        # k = b/m
        k = self.friction/self.mass
        omega = np.sqrt(-(k/2)**2 + self.g/self.length)
        q = np.exp(-k/2 * t) * (q0)

        c1 = q0
        c2 = (qd0 + q0 * k/2)/omega
        q = (c1 * np.cos(omega * t) + c2 * np.sin(omega * t)) * np.exp(-k/2 * t)
        qd = ((c2*omega - c1*k/2) * np.cos(omega*t) - (c1*omega + c2*k/2) * np.sin(omega*t)) * np.exp(-k/2 * t)

        self.trajectory = np.array([q, qd]).T

        # nt = np.array((q + q*sigma * np.random.randn(self.steps), qd + qd*sigma * np.random.randn(self.steps)))
        # X = []
        # y = []
        # for i in range(nt.T.shape[0] - horizon - 1):
        #     X.append(nt.T[i:i + 1, :].flatten())
        #     y.append(nt.T[i + 1:i + horizon + 1, :].flatten())
        self.prepare_output()
        
        # self.X = np.array(X)
        # self.y = np.array(y)
        
        return self.X, self.y, (t, self.trajectory)
    
    def plot(self, save_fig=False):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.title('Position')
        plt.ylabel('q')
        plt.xlabel('Steps i')
        plt.plot(self.X[:, 0], 'kx--')
        plt.subplot(1, 2, 2)
        plt.title('Phase Space')
        plt.plot(self.X[:, 0], self.X[:, 1], 'kx--')
        plt.xlabel('q')
        plt.ylabel('p')
        if save_fig:
            plt.savefig(self.get_filename('data'))
        plt.show()