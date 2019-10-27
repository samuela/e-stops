# Limit ourselves to single-threaded numpy operations to avoid thrashing. See
# https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# pylint: disable=wrong-import-position
import functools
from multiprocessing import Pool
from pathlib import Path
import pickle

import tqdm
import numpy as np

from research.estop.frozenlake import frozenlake
from research.estop.frozenlake import q_learning
# pylint: enable=wrong-import-position

def q_learning_job(random_seed: int, env, gamma: float,
                   policy_evaluation_frequency: int, folder: Path):
  np.random.seed(random_seed)

  states_seen, policy_rewards = q_learning.run_q_learning(
      env,
      gamma,
      policy_evaluation_frequency=policy_evaluation_frequency,
      num_episodes=50000,
      verbose=False)

  with (folder / f"seed={random_seed}.pkl").open(mode="wb") as f:
    pickle.dump({
        "states_seen": states_seen,
        "policy_rewards": policy_rewards
    }, f)

def main():
  np.random.seed(0)

  def build_env(lake: frozenlake.Lake):
    # return frozenlake.FrozenLakeEnv(lake, infinite_time=True)
    return frozenlake.FrozenLakeWithEscapingEnv(
        lake, hole_retention_probability=0.99)

  lake_map = frozenlake.MAP_8x8
  policy_evaluation_frequency = 10
  gamma = 0.99
  num_random_seeds = 96

  results_dir = Path("results/frozenlake_qlearning")
  estop_results_dir = results_dir / "estop"
  full_results_dir = results_dir / "full"
  results_dir.mkdir()
  estop_results_dir.mkdir()
  full_results_dir.mkdir()

  # Build the full environment and run value iteration to calculate the optimal
  # policy.
  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  state_action_values, _ = frozenlake.value_iteration(env,
                                                      gamma,
                                                      tolerance=1e-6)
  state_values = np.max(state_action_values, axis=-1)
  optimal_policy_reward = np.dot(state_values, env.initial_state_distribution)

  # Estimate hitting probabilities.
  optimal_policy = frozenlake.deterministic_policy(
      env, np.argmax(state_action_values, axis=-1))
  estimated_hp = frozenlake.estimate_hitting_probabilities(env,
                                                           optimal_policy,
                                                           num_rollouts=1000)
  estimated_hp2d = lake.reshape(estimated_hp)

  # Build e-stop environment.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  estop_map[estimated_hp2d <= threshold] = "E"

  estop_lake = frozenlake.Lake(estop_map)
  estop_env = build_env(estop_lake)

  # pickle dump the environemnt setup/metadata...
  pickle.dump(
      {
          "lake_map": lake_map,
          "policy_evaluation_frequency": policy_evaluation_frequency,
          "gamma": gamma,
          "num_random_seeds": num_random_seeds,
          "lake": lake,
          "env": env,
          "state_action_values": state_action_values,
          "state_values": state_values,
          "optimal_policy_reward": optimal_policy_reward,
          "optimal_policy": optimal_policy,
          "estimated_hp": estimated_hp,
          "estimated_hp2d": estimated_hp2d,
          "estop_map": estop_map,
          "percentile": percentile,
          "threshold": threshold,
          "estop_lake": estop_lake,
          "estop_env": estop_env,
      }, (results_dir / "metadata.pkl").open(mode="wb"))

  pool = Pool()

  # Run Q-learning on the full environment.
  for _ in tqdm.tqdm(pool.imap_unordered(
      functools.partial(
          q_learning_job,
          env=env,
          gamma=gamma,
          policy_evaluation_frequency=policy_evaluation_frequency,
          folder=full_results_dir,
      ), range(num_random_seeds)),
                     desc="full",
                     total=num_random_seeds):
    pass

  # Run Q-learning on the e-stop environment.
  for _ in tqdm.tqdm(pool.imap_unordered(
      functools.partial(
          q_learning_job,
          env=estop_env,
          gamma=gamma,
          policy_evaluation_frequency=policy_evaluation_frequency,
          folder=estop_results_dir,
      ), range(num_random_seeds)),
                     desc="estop",
                     total=num_random_seeds):
    pass

if __name__ == "__main__":
  main()
