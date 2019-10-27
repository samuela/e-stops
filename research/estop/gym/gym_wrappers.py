from typing import Any, NamedTuple

import gym
from gym.envs.registration import registry
from jax import random
import numpy as np

from research.statistax import Deterministic, SampleOnly
from research.estop.ddpg import Env

def openai_gym_env(gym_env, reward_adjustment: float = 0.0) -> Env:
  """A correct, safer wrapper of an OpenAI gym environment."""
  def init(rng):
    gym_env.seed(int(random.randint(rng, (), 0, 1e6)))
    return gym_env.reset()

  observed_rewards = {}

  def step(state, action):
    # Assert that state matches the current state of gym_env.
    # pylint: disable=protected-access
    assert np.allclose(state, gym_env.env._get_obs())

    obs_before = state
    obs_after, reward, _done, _info = gym_env.step(action)
    observed_rewards[(str(obs_before), str(action),
                      str(obs_after))] = reward + reward_adjustment
    return Deterministic(obs_after)

  def reward(s1, a, s2):
    # We make the assumption that we only ever calculate rewards on transitions
    # that we've already seen and added to `observed_rewards`.
    return observed_rewards[(str(s1), str(a), str(s2))]

  return Env(SampleOnly(init), step, reward)

def unsafe_openai_gym_env(gym_env, reward_adjustment: float = 0.0) -> Env:
  def init(rng):
    gym_env.seed(int(random.randint(rng, (), 0, 1e6)))
    return gym_env.reset()

  last_reward = [0.0]

  def step(_, action):
    obs_after, reward, _done, _info = gym_env.step(action)
    last_reward[0] = reward + reward_adjustment

    return Deterministic(obs_after)

  def reward(_s1, _a, _s2):
    # We make the assumption that we only ever calculate rewards based on the
    # very last transition seen.
    return last_reward[0]

  return Env(SampleOnly(init), step, reward)

class GymEnvSpec(NamedTuple):
  env_name: str
  max_episode_steps: int
  env: Env
  gym_env: Any
  state_shape: Any
  action_shape: Any

def build_env_spec(env_name: str, reward_adjustment: float) -> GymEnvSpec:
  gym_env = gym.make(env_name)
  env = unsafe_openai_gym_env(gym_env, reward_adjustment=reward_adjustment)
  return GymEnvSpec(
      env_name=env_name,
      max_episode_steps=registry.env_specs[env_name].max_episode_steps,
      env=env,
      gym_env=gym_env,
      state_shape=gym_env.observation_space.shape,
      action_shape=gym_env.action_space.shape,
  )
