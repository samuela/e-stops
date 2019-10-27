"""All of the shared environemnt parameters and configuration."""

from research.estop.pendulum.env import cost, pendulum_environment

gamma = 0.999
episode_length = 1000
max_speed = 25
max_torque = 0.5
reward_adjustment = cost([0, max_speed], [max_torque])

env = pendulum_environment(
    mass=0.1,
    length=1.0,
    gravity=9.8,
    friction=0,
    max_speed=max_speed,
    dt=0.05,
    reward_adjustment=reward_adjustment,
)

state_shape = (4, )
action_shape = (1, )
