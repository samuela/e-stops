"""Rollout an example trajectory from the pendulum environment as configured and
visualize the result."""

from jax import random
import jax.numpy as jp

from research.statistax import Deterministic
from research.estop.pendulum.config import env
from research.estop.pendulum.env import viz_pendulum_rollout
from research.estop.mdp import rollout_from_state

states, actions = rollout_from_state(
    random.PRNGKey(0),
    env,
    lambda _: Deterministic(jp.array([0.0])),
    num_timesteps=1000,
    state=jp.array([jp.pi - 0.1, 0.0, 0.0, 0.0]),
)

viz_pendulum_rollout(states, actions)
