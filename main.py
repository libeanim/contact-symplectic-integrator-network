# %% Init
import os
from experiments import pendulum, pendulum_friction, bouncing_ball, newton_cradle

# Create path for images
try:
    if not os.path.exists('images'):
        os.mkdir('images')
except:
    print('Unable to create image folder.')
# %% Pendulum
pendulum.run(train_vin=True)

pendulum.plot_trajectory(savefig=True)
pendulum.plot_energy(savefig=True)
pendulum.plot_potential(savefig=True)

# %% Bouncing Ball
bouncing_ball.run()

bouncing_ball.plot_trajectory(savefig=True)
bouncing_ball.plot_potential(savefig=True)

# %% Newton Cradle
newton_cradle.run()

newton_cradle.plot_trajectory(savefig=True)
newton_cradle.plot_potential(savefig=True)
