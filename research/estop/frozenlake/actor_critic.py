from typing import Optional

import numpy as np

from research.estop.frozenlake import frozenlake
from research.estop.frozenlake import utils

def actor_critic_episode(env,
                         gamma: float,
                         actor_optimizer,
                         critic_optimizer,
                         max_episode_length: Optional[int] = None):
  # Start off by sampling an initial state from the initial_state distribution.
  current_state = np.random.choice(env.lake.num_states,
                                   p=env.initial_state_distribution)
  episode = []

  actor_grad = np.zeros((env.lake.num_states, frozenlake.NUM_ACTIONS))
  critic_grad = np.zeros((env.lake.num_states, ))

  t = 0
  while (max_episode_length is None) or (max_episode_length is not None
                                         and t < max_episode_length):
    # Take a step.
    action_probs = utils.softmax(actor_optimizer.get()[current_state, :],
                                 axis=-1)
    action = np.random.choice(frozenlake.NUM_ACTIONS, p=action_probs)
    next_state = np.random.choice(env.lake.num_states,
                                  p=env.transitions[current_state, action, :])
    reward = env.rewards[current_state, action, next_state]

    v = critic_optimizer.get()
    delta = reward + gamma * v[next_state] - v[current_state]

    # Calculate gradients
    actor_grad[:, :] = 0.0
    actor_grad[current_state, :] -= action_probs
    actor_grad[current_state, action] += 1.0
    actor_grad *= delta * (gamma**t)

    critic_grad[:] = 0.0
    critic_grad[current_state] = delta

    actor_optimizer.step(-actor_grad)
    critic_optimizer.step(-critic_grad)

    # Continue...
    episode.append((current_state, action, reward))
    current_state = next_state
    t += 1

    if current_state in env.terminal_states:
      break

  return episode, current_state

def run_actor_critic(env,
                     gamma: float,
                     actor_optimizer,
                     critic_optimizer,
                     num_episodes: int,
                     policy_evaluation_frequency: int = 10,
                     verbose: bool = True):
  # We use this to warm start iterative policy evaluation.
  V = None

  states_seen = 0
  states_seen_log = []
  policy_rewards_log = []
  for episode_num in range(num_episodes):
    episode, _ = actor_critic_episode(env,
                                      gamma,
                                      actor_optimizer,
                                      critic_optimizer,
                                      max_episode_length=None)
    # print(f"episode length {len(episode)}")
    states_seen += len(episode)

    if episode_num % policy_evaluation_frequency == 0:
      policy = utils.softmax(actor_optimizer.get(), axis=-1)
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
        # print(env.lake.reshape(critic_optimizer.get()))

      states_seen_log.append(states_seen)
      policy_rewards_log.append(policy_reward)

  return states_seen_log, policy_rewards_log
