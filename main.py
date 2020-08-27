# %% Init
from experiments import pendulum, pendulum_friction, bouncing_ball, newton_cradle
# %% Pendulum
pendulum.run(train_vin=True)

pendulum.plot_trajectory()
pendulum.plot_energy()
pendulum.plot_potential()

# %% Pendulum friction
pendulum_friction.run(train_vin=True)

pendulum_friction.plot_trajectory()
pendulum_friction.plot_energy()
pendulum_friction.plot_potential()

# %% Bouncing Ball
bouncing_ball.run()

bouncing_ball.plot_trajectory()
bouncing_ball.plot_potential()

# %% Newton Cradle
newton_cradle.run()

newton_cradle.plot_trajectory()
newton_cradle.plot_potential()
