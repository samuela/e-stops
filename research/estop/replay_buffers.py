from typing import NamedTuple

import numpy as np
from jax import ops, random
import jax.numpy as jp

class ReplayBuffer:
  @property
  def buffer_size(self):
    raise NotImplementedError()

  def add(self, state, action, reward, next_state, done):
    raise NotImplementedError()

  def minibatch(self, rng, batch_size: int):
    raise NotImplementedError()

class JaxReplayBuffer(ReplayBuffer, NamedTuple):
  states: jp.ndarray
  actions: jp.ndarray
  rewards: jp.ndarray
  next_states: jp.ndarray
  done: jp.ndarray
  size: int

  @property
  def buffer_size(self):
    return self.states.shape[0]

  def add(self, state, action, reward, next_state, done):
    ix = self.size % self.buffer_size
    return JaxReplayBuffer(
        states=ops.index_update(self.states, ix, state),
        actions=ops.index_update(self.actions, ix, action),
        rewards=ops.index_update(self.rewards, ix, reward),
        next_states=ops.index_update(self.next_states, ix, next_state),
        done=ops.index_update(self.done, ix, done),
        size=self.size + 1,
    )

  def minibatch(self, rng, batch_size: int):
    ixs = random.randint(
        rng,
        (batch_size, ),
        minval=0,
        maxval=jp.minimum(self.buffer_size, self.size),
    )
    return (
        self.states[ixs, ...],
        self.actions[ixs, ...],
        self.rewards[ixs, ...],
        self.next_states[ixs, ...],
        self.done[ixs, ...],
    )

class NumpyReplayBuffer(ReplayBuffer):
  states: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  next_states: np.ndarray
  done: np.ndarray
  size: int

  def __init__(self, buffer_size, state_shape, action_shape):
    self.states = np.zeros((buffer_size, ) + state_shape)
    self.actions = np.zeros((buffer_size, ) + action_shape)
    self.rewards = np.zeros((buffer_size, ))
    self.next_states = np.zeros((buffer_size, ) + state_shape)
    self.done = np.zeros((buffer_size, ), dtype=bool)
    self.size = 0

  @property
  def buffer_size(self):
    # pylint: disable=unsubscriptable-object
    return self.states.shape[0]

  def add(self, state, action, reward, next_state, done):
    ix = self.size % self.buffer_size
    self.states[ix, ...] = state
    self.actions[ix, ...] = action
    self.rewards[ix, ...] = reward
    self.next_states[ix, ...] = next_state
    self.done[ix, ...] = done
    self.size += 1
    return self

  def minibatch(self, rng, batch_size: int):
    # Disable this for speeeeeeed. Don't forget to set the numpy random seed in
    # the beginning.
    # np.random.seed(int(random.randint(rng, (), 0, 1e6)))

    ixs = np.random.randint(
        low=0,
        high=np.minimum(self.buffer_size, self.size),
        size=(batch_size, ),
    )
    return (
        self.states[ixs, ...],
        self.actions[ixs, ...],
        self.rewards[ixs, ...],
        self.next_states[ixs, ...],
        self.done[ixs, ...],
    )
