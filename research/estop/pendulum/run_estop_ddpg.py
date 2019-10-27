import pickle

from jax import lax, random, vmap
import jax.numpy as jp

from research.estop import mdp
from research.estop.pendulum import config, run_ddpg

experiment_folder = "9_1ff35d1_ddpg_pendulum"
num_support_set_rollouts = 100
epsilon = 0.5
max_squared_dist = epsilon**2

def load_best_seed():
  experiment_metadata = pickle.load(
      open(f"results/{experiment_folder}/metadata.pkl", "rb"))
  num_random_seeds = experiment_metadata["num_random_seeds"]

  data = [
      pickle.load(
          open(f"results/{experiment_folder}/stuff/seed={seed}.pkl", "rb"))
      for seed in range(num_random_seeds)
  ]
  final_policy_values = jp.array(
      [x["policy_value_per_episode"][-1] for x in data])
  best_seed = int(jp.argmax(final_policy_values))
  return data[best_seed]

def build_support_set(rng, actor_params):
  def one_rollout(rollout_rng):
    states, _, _ = mdp.rollout(rollout_rng, config.env,
                               run_ddpg.policy(actor_params),
                               config.episode_length)
    return states

  return vmap(one_rollout)(random.split(rng, num_support_set_rollouts))

def main():
  rng = random.PRNGKey(0)
  num_episodes = 10000

  print(f"Loading best seed from {experiment_folder}... ", end="")
  best_seed_data = load_best_seed()
  print("done")

  print("Building support set... ", end="")
  rng, ss_rng = random.split(rng)
  actor_params, _ = best_seed_data["final_params"]

  support_set = build_support_set(ss_rng, actor_params)
  support_set_flat = jp.reshape(support_set, (-1, support_set.shape[-1]))

  # theta_min = jp.min(support_set_flat[:, 0]) - epsilon
  # theta_max = jp.max(support_set_flat[:, 0]) + epsilon
  # theta_dot_min = jp.min(support_set_flat[:, 1]) - epsilon
  # theta_dot_max = jp.max(support_set_flat[:, 1]) + epsilon
  print("done")

  rng, train_rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  train_reward_per_episode = []
  policy_value_per_episode = []
  episode_lengths = []

  def callback(info):
    episode = info['episode']
    reward = info['reward']

    current_actor_params = info["optimizer"].value[0]
    policy_value = run_ddpg.eval_policy(callback_rngs[episode],
                                        current_actor_params)

    print(f"Episode {episode}, "
          f"episode_length = {info['episode_length']}, "
          f"reward = {reward}, "
          f"policy_value = {policy_value}, "
          f"elapsed = {info['elapsed']}")

    train_reward_per_episode.append(reward)
    policy_value_per_episode.append(policy_value)
    episode_lengths.append(info["episode_length"])

    # if episode == num_episodes - 1:
    # if episode % 5000 == 0 or episode == num_episodes - 1:
    #   for rollout in range(5):
    #     states, actions, _ = ddpg.rollout(
    #         random.fold_in(callback_rngs[episode], rollout),
    #         config.env,
    #         policy(current_actor_params),
    #         num_timesteps=250,
    #     )
    #     viz_pendulum_rollout(states, 2 * actions / config.max_torque)

  run_ddpg.train(
      train_rng,
      num_episodes,
      # lambda t, s: lax.bitwise_or(
      #     lax.ge(t, config.episode_length),
      #     lax.bitwise_or(
      #         lax.le(s[0], theta_min),
      #         lax.bitwise_or(
      #             lax.ge(s[0], theta_max),
      #             lax.bitwise_or(lax.le(s[1], theta_dot_min),
      #                            lax.ge(s[1], theta_dot_max))))),
      # lambda t, s: lax.bitwise_or(
      #     lax.ge(t, config.episode_length),
      #     lax.bitwise_or(lax.ge(jp.abs(s[1]), 10.0),
      #                    lax.ge(jp.abs(s[0] - jp.pi), 0.5))),
      lambda loop_state: lax.bitwise_or(
          lax.ge(loop_state.episode_length, config.episode_length),
          lax.ge(
              jp.min(
                  jp.sum((support_set_flat[:, :2] - loop_state.state[:2])**2,
                         axis=1)), max_squared_dist)),
      callback,
  )

if __name__ == "__main__":
  main()
