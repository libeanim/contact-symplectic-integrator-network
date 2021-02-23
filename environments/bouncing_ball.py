import numpy as np
import matplotlib.pyplot as plt
import pymunk
from environments.base import Environment

class BouncingBall(Environment):
    """
    Bouncing Ball Environment
    =========================

    It uses the pymunk backend to generate data based on
    a Runge-Kutta method.

    Parameters:
    -----------

        steps:      int;
                    number of time steps (default: 500)
        dt:         float;
                    time steps size (default 0.02)
        epochs:     int;
                    number of epochs (default: 3000)
        CONTACT:    str;
                    contact type (default: nomax)
        mass:       float;
                    mass of the ball (default: 1.0)
        g:          float;
                    accelaration constant (default: -9.81)
        height:     float;
                    drop height
        e:          np.array;
                    Newton restitution coefficient
        horizon:    int;
                    prediction time window
        SIGMA:      float;
                    variance of gaussian white noise
        SEED:       int;
                    random seed
    """

    def __init__(
            self, steps=500, dt=0.02, epochs=3000, CONTACT='nomax', mass=1, g=-9.81,
            height=10, e=1., horizon=10, SIGMA=0., SEED=0):
        DATA = 'bouncing_ball' if e == 1. else 'bouncing_ball_zenos_paradox'
        super().__init__(DATA, CONTACT, steps, dt, horizon, SIGMA, SEED, epochs)
        self.e = e
        self.height = height
        self.mass = mass
        self.g = g

    def plot(self):
        """Plot generated data"""
        plt.plot(self.X[:, 0])
        plt.xlabel('steps')
        plt.ylabel('height')
        plt.show()

    def energy(self, data=None):
        if data is None:
            data = self.trajectory
        T = 0.5 * self.mass * data[:, 1]**2
        V = self.g * self.mass * data[:, 0]
        return T + V

    def generate(self):
        """Generate data based on parameters set in init function"""
        space = pymunk.Space()
        space.gravity = (0.0, self.g)

        radius = 0.01
        inertia = pymunk.moment_for_circle(self.mass, 0, radius, (0, 0))
        body = pymunk.Body(self.mass, inertia)
        body.position = (0, self.height)
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = self.e
        shape.friction = 0

        space.add(body, shape)
        ball = shape

        static_body = space.static_body
        static_lines = [pymunk.Segment(static_body, (-10, -1), (10, -1), 1)]

        for line in static_lines:
            # line.elasticity = 0.95
            line.friction = 0
            line.elasticity = 1
        space.add(static_lines)

        trajectory = []
        prev_velocity = ball.body.velocity[1]
        for _ in range(self.steps + 1):
            current_velocity = ball.body.velocity[1]
            delta_v = current_velocity - prev_velocity
            prev_velocity = current_velocity
            
            # Determine if a contact happened
            contact = 1. if delta_v > self.e * np.abs(prev_velocity) else 0.
            
            trajectory.append((ball.body.position[1], current_velocity, contact))
            space.step(self.dt)
        
        self.trajectory = np.array(trajectory)
        self.prepare_output()

        return self.trajectory, self.X, self.y, self.c