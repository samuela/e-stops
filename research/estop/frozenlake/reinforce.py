from typing import Optional

import numpy as np

from research.estop.frozenlake import frozenlake
from research.estop.frozenlake import utils

def reinforce_episode(env,
                      gamma: float,
                      optimizer,
                      max_episode_length: Optional[int] = None):
  raw_policy = utils.softmax(optimizer.get(), axis=-1)
  # epsilon_greedy_policy = 0.9 * raw_policy + 0.1 * np.ones(
  #     (env.lake.num_states, frozenlake.NUM_ACTIONS)) / frozenlake.NUM_ACTIONS
  episode, final_state = frozenlake.rollout(
      env, policy=raw_policy, max_episode_length=max_episode_length)
  weighted_rewards = [(gamma**t) * r for t, (_, _, r) in enumerate(episode)]

  # pylint: disable=line-too-long
  # See https://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array.
  Gs = np.cumsum(weighted_rewards[::-1])[::-1]

  grad = np.zeros((env.lake.num_states, frozenlake.NUM_ACTIONS))

  for t, (state, action, _) in enumerate(episode):
    # Do this in-place for speeeeeed!
    grad[:, :] = 0.0
    grad[state, :] -= utils.softmax(optimizer.get()[state, :])
    grad[state, action] += 1.0
    grad *= Gs[t]

    optimizer.step(-grad)

  return episode, final_state

def run_reinforce(env,
                  gamma: float,
                  optimizer,
                  num_episodes: int,
                  policy_evaluation_frequency: int = 10,
                  verbose: bool = True):
  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(num_episodes):
    episode, _ = reinforce_episode(env,
                                   gamma,
                                   optimizer,
                                   max_episode_length=None)
    # print(f"episode length {len(episode)}")
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = utils.softmax(optimizer.get(), axis=-1)
      V, _ = frozenlake.iterative_policy_evaluation(
          env,
          gamma,
          policy,
          tolerance=1e-6,
          init_V=V,
      )
      policy_reward = np.dot(V, env.initial_state_distribution)

      if verbose:
        print(f"Episode {episode_num}, policy reward: {policy_reward}")
        # print(optimizer.get())
        # print(utils.softmax(optimizer.get(), axis=-1))
        # print(1.0 * env.lake.reshape(
        #     np.argmax(policy, axis=-1) == deleteme_opt_policy))

      states_seen_log.append(states_seen)
      policy_rewards_log.append(policy_reward)

    # if (episode_num + 1) % 1000 == 0:
    #   plt.figure()
    #   viz.plot_heatmap(env, V)
    #   plt.title(f"Episode {episode_num}")
    #   plt.show()

  return states_seen_log, policy_rewards_log
