from typing import Optional

import numpy as np

from research.estop.frozenlake import frozenlake

def epsilon_greedy(epsilon: float):
  def h(action_values, _):
    # With prob. epsilon we pick a non-greedy action uniformly at random. There
    # are NUM_ACTIONS - 1 non-greedy actions.
    p = epsilon / (frozenlake.NUM_ACTIONS - 1) * np.ones(action_values.shape)
    p[np.argmax(action_values)] = 1 - epsilon
    return p

  return h

def epsilon_greedy_annealed(epsilon: float):
  def h(action_values, t: int):
    # With prob. epsilon we pick a non-greedy action uniformly at random. There
    # are NUM_ACTIONS - 1 non-greedy actions.
    p = epsilon / (t + 1) / (frozenlake.NUM_ACTIONS - 1) * np.ones(
        action_values.shape)
    p[np.argmax(action_values)] = 1 - epsilon / (t + 1)
    return p

  return h

def q_learning_episode(env,
                       gamma: float,
                       alpha: float,
                       Q,
                       meta_policy,
                       max_episode_length: Optional[int] = None):
  # Start off by sampling an initial state from the initial_state distribution.
  current_state = np.random.choice(env.lake.num_states,
                                   p=env.initial_state_distribution)
  episode = []

  t = 0
  while (max_episode_length is None) or (max_episode_length is not None
                                         and t < max_episode_length):
    # Assert that action_probs is not None in order to avoid a pernicious set of
    # bugs where the meta_policy forgets a return statement.
    action_probs = meta_policy(Q[current_state, :], t)
    assert action_probs is not None

    action = np.random.choice(frozenlake.NUM_ACTIONS, p=action_probs)
    next_state = np.random.choice(env.lake.num_states,
                                  p=env.transitions[current_state, action, :])
    reward = env.rewards[current_state, action, next_state]

    Q[current_state, action] += alpha * (
        reward + gamma * Q[next_state, :].max() - Q[current_state, action])

    episode.append((current_state, action, reward))
    current_state = next_state
    t += 1

    if current_state in env.terminal_states:
      break

  # `current_state` is now the final state. Reporting it is necessary in order
  # to tell which state the episode actually ended on.
  return Q, episode, current_state

def run_q_learning(
    env: frozenlake.FrozenLakeEnv,
    gamma: float,
    num_episodes: int,
    policy_evaluation_frequency: int = 10,
    verbose: bool = True,
):
  # Initializing to random values is necessary to break ties, preventing the
  # agent from always picking the same action and never getting anywhere.
  Q = np.random.rand(env.lake.num_states, frozenlake.NUM_ACTIONS)

  # This is crucial! There is no positive or negative reward for taking any
  # action in a terminal state. See Sutton & Barto page 131.
  for s in env.terminal_states:
    # For the life of me, I don't understand why this disable is necessary. It
    # only seems necessary on circleci, even though the pylint version there is
    # exactly same as locally.
    # pylint: disable=unsupported-assignment-operation
    Q[s, :] = 0.0

  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(num_episodes):
    Q, episode, _ = q_learning_episode(
        env,
        gamma,
        alpha=0.1,
        Q=Q,
        meta_policy=epsilon_greedy(epsilon=0.1),
        # meta_policy=epsilon_greedy_annealed(epsilon=1.0),
        max_episode_length=None)
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = frozenlake.deterministic_policy(env, np.argmax(Q, axis=-1))
      V, _ = frozenlake.iterative_policy_evaluation(env,
                                                    gamma,
                                                    policy,
                                                    tolerance=1e-6,
                                                    init_V=V)
      policy_reward = np.dot(V, env.initial_state_distribution)

      if verbose:
        print(f"Episode {episode_num}, policy reward: {policy_reward}")

      states_seen_log.append(states_seen)
      policy_rewards_log.append(policy_reward)

    # if (episode_num + 1) % 1000 == 0:
    #   V = np.max(Q, axis=-1)
    #   plt.figure()
    #   viz.plot_heatmap(env, V)
    #   plt.title(f"Episode {episode_num}")
    #   plt.show()

  return states_seen_log, policy_rewards_log
