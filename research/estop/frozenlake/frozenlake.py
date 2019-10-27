# See https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

from typing import Tuple, Optional

import numpy as np

State = int
Action = int

# Order is important here because the state transitions rely on +/- 1 mod 4 to
# calculate the next state.
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

NUM_ACTIONS = 4

MAP_CORRIDOR_3x1 = np.array([["S", "F", "G"]])
MAP_CORRIDOR_4x1 = np.array([["S", "F", "F", "G"]])
MAP_4x4 = np.array([["S", "F", "F", "F"], ["F", "H", "F", "H"],
                    ["F", "F", "F", "H"], ["H", "F", "F", "G"]])
MAP_8x8 = np.array([["S", "F", "F", "F", "F", "F", "F", "F"],
                    ["F", "F", "F", "F", "F", "F", "F", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "F"],
                    ["F", "F", "F", "F", "F", "H", "F", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "F"],
                    ["F", "H", "H", "F", "F", "F", "H", "F"],
                    ["F", "H", "F", "F", "H", "F", "H", "F"],
                    ["F", "F", "F", "H", "F", "F", "F", "G"]])

XMAP_9x9 = np.array([["H", "F", "F", "F", "F", "F", "F", "F", "H"],
                     ["F", "H", "F", "F", "F", "F", "F", "H", "F"],
                     ["F", "F", "H", "F", "F", "F", "H", "F", "F"],
                     ["F", "F", "F", "H", "F", "H", "F", "F", "F"],
                     ["F", "F", "F", "F", "S", "F", "F", "F", "G"],
                     ["F", "F", "F", "H", "F", "H", "F", "F", "F"],
                     ["F", "F", "H", "F", "F", "F", "H", "F", "F"],
                     ["F", "H", "F", "F", "F", "F", "F", "H", "F"],
                     ["H", "F", "F", "F", "F", "F", "F", "F", "H"]])

class Lake:
  def __init__(self, lake_map):
    self.lake_map = lake_map

    self.width, self.height = self.lake_map.shape
    self.num_states = self.width * self.height
    self.ij_states = [(i, j) for i in range(self.width)
                      for j in range(self.height)]

    self.estop_states = [
        si for si, (i, j) in enumerate(self.ij_states)
        if self.lake_map[i, j] == "E"
    ]
    self.goal_states = [
        si for si, (i, j) in enumerate(self.ij_states)
        if self.lake_map[i, j] == "G"
    ]
    self.hole_states = [
        si for si, (i, j) in enumerate(self.ij_states)
        if self.lake_map[i, j] == "H"
    ]
    self.frozen_states = [
        si for si, (i, j) in enumerate(self.ij_states)
        if self.lake_map[i, j] == "F"
    ]

    ss = [
        si for si, (i, j) in enumerate(self.ij_states)
        if self.lake_map[i, j] == "S"
    ]
    assert len(ss) == 1
    self.start_state = ss[0]

  def reshape(self, stuff1d):
    stuff2d = np.zeros((self.width, self.height), dtype=stuff1d.dtype)
    for s, v in zip(self.ij_states, stuff1d):
      stuff2d[s] = v
    return stuff2d

  def _clip(self, pseudo_state: Tuple[int, int]) -> State:
    i, j = pseudo_state
    return self.ij_states.index((
        np.clip(i, 0, self.width - 1),
        np.clip(j, 0, self.height - 1),
    ))

  def move(self, state: State, action: Action) -> State:
    i, j = self.ij_states[state]
    if action == LEFT:
      return self._clip((i, j - 1))
    elif action == DOWN:
      return self._clip((i + 1, j))
    elif action == RIGHT:
      return self._clip((i, j + 1))
    elif action == UP:
      return self._clip((i - 1, j))
    else:
      raise Exception("bad action")

class FrozenLakeEnv:
  lake: Lake
  infinite_time: bool

  def __init__(self, lake: Lake, infinite_time: bool):
    self.lake = lake
    self.infinite_time = infinite_time

    self.initial_state_distribution = np.zeros((self.lake.num_states, ))
    self.initial_state_distribution[self.lake.start_state] = 1.0

    # E-stop states are always terminal. The hole and goal states are terminal
    # iff the environment is finite-time.
    self.terminal_states = self.lake.estop_states + (
        self.lake.goal_states +
        self.lake.hole_states if not self.infinite_time else [])
    self.nonterminal_states = [
        i for i in range(self.lake.num_states) if i not in self.terminal_states
    ]

    self.transitions = FrozenLakeEnv.build_transitions(self.lake)
    self.rewards = FrozenLakeEnv.build_rewards(self.lake, self.infinite_time)

  @staticmethod
  def build_transitions(lake: Lake):
    transitions = np.zeros((lake.num_states, NUM_ACTIONS, lake.num_states))
    for s in range(lake.num_states):
      if lake.lake_map[lake.ij_states[s]] in ["E", "H", "G"]:
        transitions[s, :, s] = 1.0
      else:
        for a in [LEFT, DOWN, RIGHT, UP]:
          # Use += instead of = in the weird situation in which two moves
          # collide.
          transitions[s, a, lake.move(s, (a - 1) % NUM_ACTIONS)] += 1.0 / 3.0
          transitions[s, a, lake.move(s, a)] += 1.0 / 3.0
          transitions[s, a, lake.move(s, (a + 1) % NUM_ACTIONS)] += 1.0 / 3.0

    return transitions

  @staticmethod
  def build_rewards(lake: Lake, infinite_time: bool):
    rewards = np.zeros((lake.num_states, NUM_ACTIONS, lake.num_states))
    for s in lake.goal_states:
      rewards[:, :, s] = 1.0

      if not infinite_time:
        # Staying in a goal state means no reward.
        rewards[s, :, s] = 0.0

    return rewards

class FrozenLakeWithEscapingEnv:
  # pylint: disable=too-few-public-methods

  lake: Lake
  hole_retention_probability: float

  def __init__(self, lake: Lake, hole_retention_probability: float):
    self.lake = lake
    self.hole_retention_probability = hole_retention_probability

    # The goal state is considered terminal.
    self.infinite_time = False

    self.initial_state_distribution = np.zeros((self.lake.num_states, ))
    self.initial_state_distribution[self.lake.start_state] = 1.0

    # E-stop and goal states are terminal. Hole states can be escaped.
    self.terminal_states = self.lake.estop_states + self.lake.goal_states
    self.nonterminal_states = [
        i for i in range(self.lake.num_states) if i not in self.terminal_states
    ]

    self.transitions = FrozenLakeWithEscapingEnv.build_transitions(
        self.lake, self.hole_retention_probability)
    self.rewards = FrozenLakeEnv.build_rewards(self.lake, infinite_time=False)

  @staticmethod
  def build_transitions(lake: Lake, hole_retention_probability: float):
    transitions = np.zeros((lake.num_states, NUM_ACTIONS, lake.num_states))
    for s in range(lake.num_states):
      if lake.lake_map[lake.ij_states[s]] in ["E", "G"]:
        transitions[s, :, s] = 1.0
      elif lake.lake_map[lake.ij_states[s]] == "H":
        for a in [LEFT, DOWN, RIGHT, UP]:
          leave_prob = 1.0 / 3.0 * (1 - hole_retention_probability)

          # Stay in the same hole with probability hole_retention_probability.
          transitions[s, a, s] = hole_retention_probability

          # Otherwise we leave stochastically.
          transitions[s, a, lake.move(s, (a - 1) % NUM_ACTIONS)] += leave_prob
          transitions[s, a, lake.move(s, a)] += leave_prob
          transitions[s, a, lake.move(s, (a + 1) % NUM_ACTIONS)] += leave_prob
      else:
        for a in [LEFT, DOWN, RIGHT, UP]:
          # Use += instead of = in the weird situation in which two moves
          # collide.
          transitions[s, a, lake.move(s, (a - 1) % NUM_ACTIONS)] += 1.0 / 3.0
          transitions[s, a, lake.move(s, a)] += 1.0 / 3.0
          transitions[s, a, lake.move(s, (a + 1) % NUM_ACTIONS)] += 1.0 / 3.0

    return transitions

def expected_rewards(env: FrozenLakeEnv):
  return np.einsum("ijk,ijk->ij", env.transitions, env.rewards)
  # expected_rewards2 = np.sum(transitions * rewards, axis=-1)
  # assert np.allclose(expected_rewards, expected_rewards2)

def value_iteration(env: FrozenLakeEnv,
                    gamma: float,
                    tolerance: Optional[float] = None,
                    max_iterations: Optional[int] = None,
                    callback=lambda _: None):
  """See Sutton & Barto page 83."""
  V = np.zeros((env.lake.num_states, ))
  Q = np.zeros((env.lake.num_states, NUM_ACTIONS))

  # Seed the values of the goal states with the geometric sum, since we know
  # that's the answer analytically. This only makes sense when we allow
  # ourselves to pick up rewards staying in the goal state forever.
  if env.infinite_time:
    for s in env.lake.goal_states:
      V[s] = 1.0 / (1.0 - gamma)

  expected_r = expected_rewards(env)

  num_iterations = 0
  policy_rewards_per_iter = []
  while True:
    Q = expected_r + gamma * np.einsum("ijk,k->ij", env.transitions, V)
    new_state_values = np.max(Q, axis=-1)

    delta = np.abs(V - new_state_values).max()
    policy_reward = np.dot(new_state_values, env.initial_state_distribution)
    callback({
        "iteration": num_iterations,
        "Q": Q,
        "delta": delta,
        "policy_reward": policy_reward,
    })
    V = new_state_values
    policy_rewards_per_iter.append(policy_reward)
    num_iterations += 1

    if tolerance is not None and delta <= tolerance:
      break
    if max_iterations is not None and num_iterations >= max_iterations:
      break

  return Q, policy_rewards_per_iter

def iterative_policy_evaluation(env: FrozenLakeEnv,
                                gamma: float,
                                policy,
                                tolerance: float,
                                init_V=None):
  """See Sutton & Barto page 75."""
  if init_V is None:
    V = np.zeros((env.lake.num_states, ))

    # Seed the values of the goal states with the geometric sum, since we know
    # that's the answer analytically. This only makes sense when we allow
    # ourselves to pick up rewards staying in the goal state forever.
    if env.infinite_time:
      for s in env.lake.goal_states:
        V[s] = 1.0 / (1.0 - gamma)
  else:
    V = init_V

  expected_r = expected_rewards(env)

  delta = np.inf
  policy_rewards_per_iter = []
  while delta > tolerance:
    Q = expected_r + gamma * np.einsum("ijk,k->ij", env.transitions, V)
    new_state_values = np.einsum("ij,ij->i", Q, policy)
    delta = np.abs(V - new_state_values).max()
    policy_reward = np.dot(new_state_values, env.initial_state_distribution)
    V = new_state_values
    policy_rewards_per_iter.append(policy_reward)

  return V, policy_rewards_per_iter

def markov_chain_stats(env: FrozenLakeEnv, policy_transitions):
  assert policy_transitions.shape == (env.lake.num_states, env.lake.num_states)

  # See https://en.wikipedia.org/wiki/Absorbing_Markov_chain.
  absorbing_states = env.terminal_states
  transient_states = env.nonterminal_states
  t = len(transient_states)

  # See https://stackoverflow.com/questions/19161512/numpy-extract-submatrix.
  Q = policy_transitions[np.ix_(transient_states, transient_states)]
  R = policy_transitions[np.ix_(transient_states, absorbing_states)]
  Ninv = np.eye(t) - Q
  N = np.linalg.inv(Ninv)

  # Calculate the hitting probabilities.
  # pylint: disable=unsubscriptable-object
  transient_hp = (N - np.eye(t)) * np.power(np.diag(N), -1)[np.newaxis, :]
  absorbing_hp = np.linalg.solve(Ninv, R)

  hitting_prob = np.zeros(env.lake.num_states)
  start_ix = transient_states.index(env.lake.start_state)
  hitting_prob[transient_states] = transient_hp[start_ix, :]
  hitting_prob[absorbing_states] = absorbing_hp[start_ix, :]

  # Calculate the expected number of steps to absorption.
  esta = np.zeros(env.lake.num_states)
  esta[transient_states] = np.sum(N, axis=1)

  return hitting_prob, esta

def rollout(env, policy, max_episode_length: Optional[int] = None):
  # Start off by sampling an initial state from the initial_state distribution.
  current_state = np.random.choice(env.lake.num_states,
                                   p=env.initial_state_distribution)
  episode = []

  t = 0
  while (max_episode_length is None) or (max_episode_length is not None
                                         and t < max_episode_length):
    action = np.random.choice(NUM_ACTIONS, p=policy[current_state, :])
    next_state = np.random.choice(env.lake.num_states,
                                  p=env.transitions[current_state, action, :])
    reward = env.rewards[current_state, action, next_state]

    episode.append((current_state, action, reward))
    current_state = next_state
    t += 1

    if current_state in env.terminal_states:
      break

  # `current_state` is now the final state. Reporting it is necessary in order
  # to tell which state the episode actually ended on.
  return episode, current_state

def estimate_hitting_probabilities(env: FrozenLakeEnv, policy,
                                   num_rollouts: int):
  def rollout_states():
    episode, final_state = rollout(env, policy)
    # We skip the starting state in order to match what the analytic solution
    # computes.
    # return [s for s, _, _ in episode[1:]] + [final_state]

    return [s for s, _, _ in episode] + [final_state]

  rollouts = [rollout_states() for _ in range(num_rollouts)]
  return np.array([
      sum((s in rollout) for rollout in rollouts) / num_rollouts
      for s in range(env.lake.num_states)
  ])

def num_mdp_states(lake_map):
  num_starts = (lake_map == "S").sum()
  num_frozen = (lake_map == "F").sum()
  num_holes = (lake_map == "H").sum()
  num_goals = (lake_map == "G").sum()
  # We can reduce all holes to a single MDP state.
  return num_starts + num_frozen + (num_holes > 0) + num_goals

def deterministic_policy(env: FrozenLakeEnv, actions):
  """Convert a vector mapping states to actions to a policy array."""
  # pylint: disable=line-too-long
  # See https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
  policy = np.zeros((env.lake.num_states, NUM_ACTIONS))
  policy[np.arange(env.lake.num_states), actions] = 1.0
  return policy

def optimal_policy_reward(env, gamma: float):
  _, policy_values_per_iter = value_iteration(env, gamma, tolerance=1e-6)
  return policy_values_per_iter[-1]
