# CD-Lagrange Network


## Structure

This projects has four different modules

- `enviroments` which contains classes to simulate following environments:
    - Bouncing Ball
    - Pendulum
    - Newton Cradle
- `experiments` which contains the concrete setups used in the dissertation. This includes the parameter configuration and the code for creating the plots used in this section.
- `models` contains the implementation of integrator analysed in this work (CD-Lagrange) as well as the models it is compared to (residual network, VIN).
- `utils` contains the `TRAIN` function which universally trains a network model given a specific environment and the `PREDICT` function which can be used to predict the trajectory of a given model and initial state $$q_0$$.

## Usage

### Simulate experiments from the thesis
In order to rerun the experiments in the thesis one can just execute the `main.py` file which will sequentially execute all experiments.
It is possible to access the models trained in every experiment by using dot-access:
```python
# Get the environment object of the pendulum experiment.
pendulum.env
# Get the cdl model of the pendulum experiment.
pendulum.cdl_model

# Get the environment object of the pendulum experiment.
newton_cradle.env
# Get the cdl model of the pendulum experiment.
newton_cradle.cdl_model
```
### Create new experiments
In order to create a new experiment one can directly import the modules (in i.e. a jupyter notebook) and set the desired parameters

```python
from environments import NewtonCradle
from models import CDLNetwork, ResNet
from utils import TRAIN, PREDICT

# Initialise the environment
env = NewtonCradle(steps=700, dt=0.01, epochs=1000)
# Generate environment data
env.generate()

# Train CD-LAGRANGE
cdl_model = TRAIN(env, CDLNetwork, name='CDL')

# Train RESNET
resnet = TRAIN(env, ResNet, name='ResNet')

# Predict trajectory given the initial state q0
cdl_data = PREDICT(env, cdl_model)
resnet_data = PREDICT(env, cdl_data)
```

In order to get a unique filename for the parameters used one can facilitate the function `env.get_filename(name, suffix='png')`.
This is useful for saving plots for example.

## Acknowledgement

Thank you for Steindor, Alex and Marc for their exellent support