import functools
from multiprocessing import cpu_count, get_context
import os
from pathlib import Path
import pickle

import gym
import tqdm
import numpy as np

from research.estop.gym.ddpg_training import batch_job
from research.estop.gym.half_cheetah import env_name, reward_adjustment

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

def main():
  num_random_seeds = cpu_count() // 2
  num_episodes = 10000
  policy_evaluation_frequency = 100
  policy_video_frequency = 1000

  # Create necessary directory structure.
  results_dir = Path("results/ddpg_half_cheetah")
  results_dir.mkdir()

  pickle.dump(
      {
          "type": "vanilla",
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "policy_evaluation_frequency": policy_evaluation_frequency,
          "policy_video_frequency": policy_video_frequency,
      }, (results_dir / "metadata.pkl").open(mode="wb"))

  state_shape = gym.make(env_name).observation_space.shape

  # See https://codewithoutrules.com/2018/09/04/python-multiprocessing/.
  # Running a single job usually takes up about 1.5-2 cores since mujoco runs
  # separately and we can't really control its parallelism.
  with get_context("spawn").Pool(processes=cpu_count() // 2) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(
        functools.partial(
            batch_job,
            env_name=env_name,
            reward_adjustment=reward_adjustment,
            num_episodes=num_episodes,
            state_min=-np.inf * np.ones(state_shape),
            state_max=np.inf * np.ones(state_shape),
            out_dir=results_dir,
            policy_evaluation_frequency=policy_evaluation_frequency,
            policy_video_frequency=policy_video_frequency),
        range(num_random_seeds)),
                       total=num_random_seeds):
      pass

if __name__ == "__main__":
  main()
