from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from research.estop.frozenlake import viz

full_results_dir = Path("results/9_1ff35d1_ddpg_pendulum")
estop_results_dir = Path("results/10_58d6605_estop_ddpg_pendulum")
num_random_seeds = 72

def load_policy_values(results_dir):
  metadata = pickle.load((results_dir / "metadata.pkl").open(mode="rb"))
  assert num_random_seeds == metadata["num_random_seeds"]
  all_seeds = [
      pickle.load((results_dir / "stuff" / f"seed={seed}.pkl").open(mode="rb"))
      for seed in range(num_random_seeds)
  ]

  # The full results were not saved with episode_lengths. We should probably
  # rectify this in the future and remove this bandaid.
  episode_lengths = np.array([
      x.get("episode_lengths", [1001] * metadata["num_episodes"])
      for x in all_seeds
  ])
  policy_value_per_episode = np.array(
      [x["policy_value_per_episode"] for x in all_seeds])

  return episode_lengths, policy_value_per_episode

if __name__ == "__main__":
  print("Loading full results...")
  full_episode_lengths, full_policy_values = load_policy_values(
      full_results_dir)
  full_steps_seen = np.cumsum(full_episode_lengths, axis=-1)
  print("... done")

  print("Loading e-stop results...")
  estop_episode_lengths, estop_policy_values = load_policy_values(
      estop_results_dir)
  estop_steps_seen = np.cumsum(estop_episode_lengths, axis=-1)
  print("... done")

  num_steps_seen_to_plot = 25000
  freq = 100
  x = freq * np.arange(int(num_steps_seen_to_plot / freq))
  estop_policy_values_interp = np.array([
      np.interp(x,
                estop_steps_seen[i, :],
                estop_policy_values[i, :],
                right=estop_policy_values[i, -1])
      for i in range(num_random_seeds)
  ])
  full_policy_values_interp = np.array([
      np.interp(x,
                full_steps_seen[i, :],
                full_policy_values[i, :],
                right=full_policy_values[i, -1])
      for i in range(num_random_seeds)
  ])

  plt.rcParams.update({"font.size": 16})
  plt.figure()
  viz.plot_errorfill(x / 1000, full_policy_values_interp, "slategrey")
  viz.plot_errorfill(x / 1000, estop_policy_values_interp, "crimson")
  plt.legend(["Full env. DDPG", "E-stop DDPG"], loc="lower right")
  plt.xlabel("Timesteps (thousands)")
  plt.ylabel("Cumulative policy reward")
  plt.tight_layout()
  plt.savefig("figs/full_vs_estop_ddpg_pendulum.pdf")
