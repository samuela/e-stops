import pickle

from jax import random
import jax.numpy as jp

from research.estop.pendulum import config
from research.estop.pendulum.env import viz_pendulum_rollout
from research.estop.pendulum.run_ddpg import actor
from research.statistax import Deterministic
from research.estop.mdp import rollout

experiment_folder = "ddpg_pendulum"
experiment_metadata = pickle.load(
    open(f"results/{experiment_folder}/metadata.pkl", "rb"))
num_random_seeds = experiment_metadata["num_random_seeds"]

data = [
    pickle.load(open(f"results/{experiment_folder}/full/seed={seed}.pkl",
                     "rb")) for seed in range(num_random_seeds)
]
final_policy_values = jp.array(
    [x["policy_value_per_episode"][-1] for x in data])
best_seed = int(jp.argmax(final_policy_values))

print(f"Best seed: {best_seed}")
print(f"Policy cumulative reward: {final_policy_values[best_seed]}")

actor_params, _ = data[best_seed]["final_params"]

rng = random.PRNGKey(0)
while True:
  rollout_rng, rng = random.split(rng)
  states, actions, _ = rollout(
      rollout_rng,
      config.env,
      lambda s: Deterministic(actor(actor_params, s)),
      num_timesteps=250,
  )
  viz_pendulum_rollout(states, 2 * actions / config.max_torque)
