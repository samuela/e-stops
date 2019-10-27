import functools
from multiprocessing import get_context
import os
from pathlib import Path
import pickle

import tqdm
from jax import lax, random
import jax.numpy as jp

from research.estop.pendulum import config, run_ddpg, run_estop_ddpg

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

num_episodes = 10000

# Maybe unify this with the job in run_ddpg_batch.
def job(
    random_seed: int,
    base_dir: Path,
    theta_min: float,
    theta_max: float,
    theta_dot_min: float,
    theta_dot_max: float,
):
  rng = random.PRNGKey(random_seed)

  rng, train_rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  params = [None]
  tracking_params = [None]

  train_reward_per_episode = []
  policy_value_per_episode = []
  episode_lengths = []
  elapsed_per_episode = []

  def callback(info):
    episode = info['episode']
    params[0] = info["optimizer"].value
    tracking_params[0] = info["tracking_params"]

    policy_value = run_ddpg.eval_policy(callback_rngs[episode],
                                        info["optimizer"].value[0])

    train_reward_per_episode.append(info['reward'])
    policy_value_per_episode.append(policy_value)
    episode_lengths.append(info["episode_length"])
    elapsed_per_episode.append(info["elapsed"])

  run_ddpg.train(
      train_rng,
      num_episodes,
      lambda t, s: lax.bitwise_or(
          lax.ge(t, config.episode_length),
          lax.bitwise_or(
              lax.le(s[0], theta_min),
              lax.bitwise_or(
                  lax.ge(s[0], theta_max),
                  lax.bitwise_or(lax.le(s[1], theta_dot_min),
                                 lax.ge(s[1], theta_dot_max))))),
      callback,
  )
  with (base_dir / f"seed={random_seed}.pkl").open(mode="wb") as f:
    pickle.dump(
        {
            "final_params": params[0],
            "final_tracking_params": tracking_params[0],
            "train_reward_per_episode": train_reward_per_episode,
            "policy_value_per_episode": policy_value_per_episode,
            "episode_lengths": episode_lengths,
            "elapsed_per_episode": elapsed_per_episode,
        }, f)

def main():
  num_random_seeds = 72
  rng = random.PRNGKey(0)

  # Create necessary directory structure.
  results_dir = Path("results/estop_ddpg_pendulum")
  full_results_dir = results_dir / "estop"
  results_dir.mkdir()
  full_results_dir.mkdir()

  pickle.dump(
      {
          "gamma": config.gamma,
          "episode_length": config.episode_length,
          "max_torque": config.max_torque,
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "tau": run_ddpg.tau,
          "buffer_size": run_ddpg.buffer_size,
          "batch_size": run_ddpg.batch_size,
          "run_estop_ddpg.experiment_folder": run_estop_ddpg.experiment_folder,
          "run_estop_ddpg.num_support_set_rollouts":
          run_estop_ddpg.num_support_set_rollouts,
          "run_estop_ddpg.epsilon": run_estop_ddpg.epsilon,
      }, (results_dir / "metadata.pkl").open(mode="wb"))

  print(f"Loading best seed from {run_estop_ddpg.experiment_folder}... ",
        end="")
  best_seed_data = run_estop_ddpg.load_best_seed()
  print("done")

  print("Building support set... ", end="")
  rng, ss_rng = random.split(rng)
  actor_params, _ = best_seed_data["final_params"]

  support_set = run_estop_ddpg.build_support_set(ss_rng, actor_params)
  support_set_flat = jp.reshape(support_set, (-1, support_set.shape[-1]))

  theta_min = jp.min(support_set_flat[:, 0]) - run_estop_ddpg.epsilon
  theta_max = jp.max(support_set_flat[:, 0]) + run_estop_ddpg.epsilon
  theta_dot_min = jp.min(support_set_flat[:, 1]) - run_estop_ddpg.epsilon
  theta_dot_max = jp.max(support_set_flat[:, 1]) + run_estop_ddpg.epsilon
  print("done")

  # See https://codewithoutrules.com/2018/09/04/python-multiprocessing/.
  with get_context("spawn").Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(
        functools.partial(
            job,
            base_dir=full_results_dir,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_dot_min=theta_dot_min,
            theta_dot_max=theta_dot_max,
        ), range(num_random_seeds)),
                       desc="estop",
                       total=num_random_seeds):
      pass

if __name__ == "__main__":
  main()
