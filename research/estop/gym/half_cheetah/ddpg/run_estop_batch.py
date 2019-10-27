import functools
from multiprocessing import cpu_count, get_context
import os
from pathlib import Path
import pickle

import tqdm
from jax import random

from research.estop.gym.ddpg_training import (batch_job,
                                              make_default_ddpg_train_config,
                                              build_env_spec)
from research.estop.gym.half_cheetah import env_name, reward_adjustment
from research.estop.gym.half_cheetah.ddpg import debug_run_estop

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

output_results_dir = Path("results/estop_ddpg_half_cheetah")

num_support_set_rollouts = 128

num_random_seeds = cpu_count() // 2
num_episodes = 20000
policy_evaluation_frequency = 1000
policy_video_frequency = 1000

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  # This is assuming that we used the default train config. If not, well...
  # don't do that.
  env_spec = build_env_spec(env_name, reward_adjustment)
  train_config = make_default_ddpg_train_config(env_spec)

  expert_rollouts = debug_run_estop.run_expert_rollouts(rng)
  state_min, state_max = debug_run_estop.get_estop_bounds(expert_rollouts)

  ###

  # Create necessary directory structure.
  output_results_dir.mkdir()

  pickle.dump(
      {
          "type": "estop",
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "policy_evaluation_frequency": policy_evaluation_frequency,
          "policy_video_frequency": policy_video_frequency,

          # E-stop specific
          "num_support_set_rollouts": num_support_set_rollouts,
          "state_min": state_min,
          "state_max": state_max,
      },
      (output_results_dir / "metadata.pkl").open(mode="wb"))

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
            state_min=state_min,
            state_max=state_max,
            out_dir=output_results_dir,
            policy_evaluation_frequency=policy_evaluation_frequency,
            policy_video_frequency=policy_video_frequency),
        range(num_random_seeds)),
                       total=num_random_seeds):
      pass
