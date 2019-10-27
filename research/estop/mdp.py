from typing import NamedTuple, Callable, TypeVar

from jax import lax, random, vmap
import jax.numpy as jp

from research.statistax import Distribution

State = TypeVar("State")
Action = TypeVar("Action")

class Env(NamedTuple):
  initial_distribution: Distribution
  step: Callable[[State, Action], Distribution]
  reward: Callable[[State, Action, State], jp.ndarray]

def rollout(rng, env: Env, policy, num_timesteps: int):
  init_rng, steps_rng = random.split(rng)
  init_state = env.initial_distribution.sample(init_rng)
  return rollout_from_state(steps_rng, env, policy, num_timesteps, init_state)

def rollout_from_state(rng, env: Env, policy, num_timesteps: int, state):
  def step(state, step_rng):
    action_rng, dynamics_rng = random.split(step_rng)
    action = policy(state).sample(action_rng)
    next_state = env.step(state, action).sample(dynamics_rng)
    reward = env.reward(state, action, next_state)
    return next_state, (state, action, reward)

  _, res = lax.scan(step, state, random.split(rng, num_timesteps))
  return res

def evaluate_policy(env: Env, policy, num_timesteps: int, num_rollouts: int,
                    gamma: float):
  def one_rollout(rollout_rng, p):
    _, _, rewards = rollout(rollout_rng, env, policy(p), num_timesteps)
    return jp.dot(rewards, jp.power(gamma, jp.arange(num_timesteps)))

  def many_rollouts(rng, p):
    rngs = random.split(rng, num_rollouts)
    return jp.mean(vmap(one_rollout, in_axes=(0, None))(rngs, p))

  return many_rollouts
