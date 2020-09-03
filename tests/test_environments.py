#%%
import sys
sys.path.append('../')
from environments import Pendulum

# Simulate diffrent pendulum scenarios to determine the correctness of the system
#%%
# Test if step count is correct
test_steps = 400
test_horizon=10
e1 = Pendulum(steps=test_steps, horizon=test_horizon, dt=0.01, epochs=3000, friction=0., length=1, SIGMA=0.1)
e1.generate()
assert e1.trajectory.shape[0] == test_steps
assert e1.trajectory.shape[1] == 2
assert e1.y.shape[0] == test_steps - test_horizon - 1
assert e1.X.shape[0] == test_steps - test_horizon - 1
assert e1.X.shape[1] == 2
assert e1.y.shape[1] == 2 * test_horizon

# Test if horizon works properly
test_horizon=5
e2 = Pendulum(steps=test_steps, horizon=test_horizon, dt=0.01, epochs=3000, friction=0., length=1, SIGMA=0.1)
e2.generate()
assert e2.y.shape[0] == test_steps - test_horizon - 1
assert e2.y.shape[1] == 2 * test_horizon

print('Successfully finished.')