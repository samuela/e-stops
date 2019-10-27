from pathlib import Path
import pickle

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from research.estop.gym.ddpg_training import (deterministic_policy,
                                              make_default_ddpg_train_config,
                                              build_env_spec, rollout,
                                              debug_run)
from research.estop.gym.half_cheetah import env_name, reward_adjustment

input_results_dir = Path("results/13_ed7ee131_ddpg_half_cheetah")

num_support_set_rollouts = 128

num_episodes = 10000
policy_evaluation_frequency = 100
policy_video_frequency = 1000

# This is assuming that we used the default train config. If not, well... don't
# do that.
env_spec = build_env_spec(env_name, reward_adjustment)
train_config = make_default_ddpg_train_config(env_spec)

def run_expert_rollouts(rng) -> np.ndarray:
  print("Loading vanilla DDPG results...")
  experiment_metadata = pickle.load(
      (input_results_dir / "metadata.pkl").open("rb"))

  data = [
      pickle.load((input_results_dir / f"seed={seed}" / "data.pkl").open("rb"))
      for seed in tqdm.trange(experiment_metadata["num_random_seeds"])
  ]
  final_policy_values = np.array([x["policy_evaluations"][-1] for x in data])
  best_seed = int(np.argmax(final_policy_values))
  print(f"... best seed is {best_seed}"
        f"with cumulative reward: {data[best_seed]['policy_evaluations'][-1]}")

  print("Rolling out trajectories from best policy...")
  actor_params, _ = data[best_seed]["final_params"]
  expert_policy = deterministic_policy(train_config, actor_params)
  return np.array([
      rollout(env_spec, r, expert_policy)[0]
      for r in tqdm.tqdm(random.split(rng, num_support_set_rollouts))
  ])

def get_estop_bounds(expert_rollouts):
  # Unfortunately there are a few special cases that need to be taken into
  # account here:
  #  * Index 1 corresponds to an angle and so any multiple of 2 pi is actually
  #    equivalent. Instead of mod'ing out this effect we just prune a bit more.
  #  * Index 8 corresponds to the velocity, which we shouldn't really bound on
  #    the max end. And on the min end we cannot trim anything off since we get
  #    initialized at zero.
  percentiles = [1, 5, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  state_min = np.array([
      np.percentile(expert_rollouts[:, :, i], p)
      for i, p in enumerate(percentiles)
  ])
  state_max = np.array([
      np.percentile(expert_rollouts[:, :, i], 100 - p)
      for i, p in enumerate(percentiles)
  ])
  std = np.std(expert_rollouts, axis=(0, 1))
  return state_min - std, state_max + std

def main():
  rng = random.PRNGKey(0)

  expert_rollouts = run_expert_rollouts(rng)
  expert_rollouts_flat = np.reshape(expert_rollouts,
                                    (-1, expert_rollouts.shape[-1]))
  state_min, state_max = get_estop_bounds(expert_rollouts)

  # Debug plot!
  plt.figure()
  for i in range(17):
    ax = plt.subplot(2, 9, i + 2)
    plt.hist(expert_rollouts_flat[:, i], bins=256)
    ax.yaxis.set_ticklabels([])
    plt.title(i)
    plt.axvline(state_min[i], c="r")
    plt.axvline(state_max[i], c="r")

  plt.show()

  debug_run(env_spec,
            train_config,
            seed=0,
            state_min=state_min,
            state_max=state_max)

if __name__ == "__main__":
  main()
