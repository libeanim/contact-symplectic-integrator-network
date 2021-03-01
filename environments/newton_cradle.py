import pymunk
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util
import matplotlib.pyplot as plt
import numpy as np
from environments.base import Environment
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class NewtonCradle(Environment):
    """
    Newton Cradle Environment
    =========================

    It supports two generator backends:
    - CD-Lagrange (cdl)
        Generates data based on the CD-Lagrange integration scheme
    - Pymunk (pymunk)
        Generates data using a Runge-Kutte method.

    Parameters:
    -----------

        steps:      int;
                    number of time steps
        dt:         float;
                    time steps size
        epochs:     int;
                    number of epochs
        CONTACT:    str;
                    contact mode
        mass:       float;
                    mass of the ball
        g:          float;
                    accelaration constant
        horizon:    int;
                    prediction time window
        SIGMA:      float;
                    variance of gaussian white noise
        SEED:       int;
                    random seed
    """

    MODEL = ('cdl', 'pymunk')

    def __init__(
            self, steps=500, dt=0.01, epochs=3000, CONTACT='nomax', mass=1., g=-9.81,
            horizon=10, SIGMA=0., SEED=0):
        super().__init__('newton-cradle', CONTACT, steps, dt, horizon, SIGMA, SEED, epochs)
        self.mass = mass
        self.g = g
        self.L = np.array([1., -1.])
        self.e = np.array([[0., 1.], [1., 0.]])

    def generate(self, model=MODEL[0], state0=np.array([ 0., 0., 2, 0.])):
        """Generate data using the selected backend
        
        Parameters
        ----------
            model:  Backend that should be used ('cdl', 'pymunk').
            state0: Initial state
        """
        if model == self.MODEL[0]:
            return self.generate_cdl(state0)
        elif model == self.MODEL[1]:
            return self.generate_pymunk()
        else:
            raise ValueError(
                'Please choose an available model ({})'.format(', '.join(self.MODEL))
            )


    def get_state(self):
        """
        Returns the current state if the pymunk backend is used.
        """
        q, qdot = [], []
        for body in self.space.bodies:
            x = body.position[0]
            y = body.position[1]

            xdot = body.velocity[0]
            ydot = body.velocity[1]
            
            q.append((x, y))
            qdot.append((xdot, ydot))
        return np.array([q, qdot]).flatten()
        
    def get_polar(self, offset=-90):
        """
        Returns the polar coordinates if the pymunk backend is used.
        """
        q, qdot = [], []
        for i, body in enumerate(self.space.bodies):
        
            x = self.joints[i][0] - body.position[0]
            y = self.joints[i][1] - body.position[1]
            # r = np.sqrt(x**2 + y**2)
            tmp_q = np.arctan2(y, x) + offset/180 * np.pi

            xdot = body.velocity[0]
            ydot = body.velocity[1]
            tmp_qdot = (x*ydot - xdot * y)/(x**2 + y**2)
            
            q.append(tmp_q)
            qdot.append(tmp_qdot)
            
        return np.array([q, qdot]).flatten()

    def convert_cartesian(self, state, r=100, offset=-90):
        """
        This function converts the generalised coordinates from a given state
        into cartesian coordinates.
        """
        dim_state = len(state)
        thetas = state[:dim_state//2]
        thetadots = state[dim_state//2:]
        
        q = []
        qdot = []
        for i, (theta, thetadot) in enumerate(zip(thetas, thetadots)):
            x = self.joints[i][0] - r * np.cos(theta - offset/180 * np.pi)
            y = self.joints[i][1] - r * np.sin(theta - offset/180 * np.pi)
            q.append([x, y])
            xdot = -r * thetadot * np.sin(theta - offset/180 * np.pi)
            ydot = r * thetadot * np.cos(theta - offset/180 * np.pi)
            qdot.append([xdot, ydot])
        return np.array([q, qdot]).flatten()

    def plot_c(self, state, convert=False):
        """
        Plots the position of the two balls for a given state.
        (This is fur debugging purposes only)
        """
        if convert:
            state = self.convert_cartesian(state)
        fig = plt.figure()
        print('Position', (state[0], state[1]), (state[2], state[3]))
        c1 = plt.Circle((state[0], state[1]), radius=10)
        c2 = plt.Circle((state[2], state[3]), radius=10)
        ax=fig.gca()
        ax.add_patch(c1)
        ax.add_patch(c2)
        plt.axis([-100, 100, -25, 100])
        plt.show()

    def generate_pymunk(self, radius=100, elasticity=1., damping=1., g=-9.81,
            balls=[(-10, 0), (10, 0)], joints=[(-10, 1000), (10, 1000)], start_impuls=-20,
            coordinate_system='polar'):
        """Generate data using the pymunk backend"""

        # Space
        space = pymunk.Space()
        space.gravity = 0,g
        space.damping = 1
        self.space = space
        self.joints = joints


        # balls
        for i in range(len(joints)):
            mass = 1
            moment = pymunk.moment_for_circle(mass, 0, radius, (0,0))
            body = pymunk.Body(mass, moment)
            body.position = balls[i]
            body.start_position = Vec2d(body.position)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 1
            space.add(body, shape)
            pj = pymunk.PinJoint(space.static_body, body, joints[i], (0,0))
            space.add(pj)
            
        space.bodies[0].apply_impulse_at_local_point((start_impuls,0))
        space.bodies[0].velocity

        trajectory = []
        for i in range(self.steps + 1):
            if coordinate_system == 'polar':
                state = self.get_polar()
            else:
                state = self.get_state()
            trajectory.append(state)
            space.step(self.dt)
        self.trajectory = np.array(trajectory)
        X, y = self.prepare_output()

        return trajectory, X, y

    def generate_cdl(self, state0=np.array([ 0., 0., 2, 0.]), enforce_constraints=True): 
        """Generate data using the CD-Lagrange integrator backend"""       
        dim_Q = 2
        self.joints = [(0,100), (0,100)]
        trajectory = [np.hstack([state0, 0])]
        M_inv = np.eye(dim_Q)
        for _ in range(self.steps):
            u = trajectory[-1][:dim_Q]
            udot = trajectory[-1][dim_Q:-1]

            u_next = u + self.dt * udot
            dUdu = self.g * np.sin(u_next)
            w = M_inv @ (self.dt * dUdu)


            # Contact forces
            i=0
            contact = u[0] - u[1] <= 0. and not udot[0] > 0 and not udot[1] < 0
            if contact:
                v = -self.e @ (self.L * udot)
                r = (v - self.L * (udot + w))
                i = M_inv @ (self.L * r)
                if enforce_constraints:
                    u_next = np.array([0, 0])

            udot_next = udot + w + i
            trajectory.append(np.hstack([np.array([u_next, udot_next]).flatten(), 1. if contact else 0.]))


        self.trajectory = np.array(trajectory)
        self.prepare_output()

        return self.X, self.y, self.trajectory

    def plot(self, savefig=False):
        """Plot generated data
        
        Parameters
        ----------

            savefig:    bool;
                        save the figure on disc
        """
        plt.plot(self.X[:, 0], self.X[:, 2], 'x--', label='1')
        plt.plot(self.X[:, 1], self.X[:, 3], 'x--', label='2')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('Angle')
        plt.plot(self.X[:, 0], label='1'); plt.plot(self.X[:, 1], label='2')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.X[:, 2], label='1'); plt.plot(self.X[:, 3], label='2')
        plt.title('Velocity')
        plt.legend()
        plt.show()
    
    def animate(self, mode='jupyter'):
        """
        Create an animation of the environment
        
        Parameters
        ----------
            mode:   'jupyter', 'save', 'else';
                    Select animation mode:
                        - jupyter: will create animation in a jupyter notebook
                        - save: Save the animation in a mp4 file
                        - else: will just show the plot (works in a qt console environment)
        """
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        # Amount of frames that should be skipped
        data_skip = 5

        tmp = []
        for i in range(self.trajectory.shape[0]):
            tmp.append(self.convert_cartesian(self.trajectory[i], r=100))
        tmp = np.array(tmp)

        def init_func():
            ax.clear()
            # plt.xlim((x[0], x[-1]))
            plt.xlim((-100, 100))
            plt.ylim((-1, 100))


        def update_plot(i):
            ax.plot(tmp[i:i+data_skip, 0], tmp[i:i+data_skip, 1], color='k', alpha=0.2)
            ax.scatter(tmp[i, 0], tmp[i, 1], marker='o', color='b', alpha=0.2)
            ax.plot(tmp[i:i+data_skip, 2], tmp[i:i+data_skip, 3], color='k', alpha=0.2)
            ax.scatter(tmp[i, 2], tmp[i, 3], marker='o', color='r', alpha=0.2)
            
        anim = FuncAnimation(fig,
                            update_plot,
                            frames=np.arange(0, tmp.shape[0], data_skip),
                            init_func=init_func,
                            interval=100)

        if mode == 'jupyter':
            HTML(anim.to_html5_video())
        elif mode == 'save':
            anim.save(self.get_filename('animation', suffix='mp4'))
        else:
            plt.show()