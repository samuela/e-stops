import pickle

import matplotlib.pyplot as plt
from jax import jit, random, vmap
import jax.numpy as jp

from research.estop import mdp
from research.estop.pendulum import config
from research.estop.pendulum import run_ddpg

experiment_folder = "9_1ff35d1_ddpg_pendulum"
num_rollouts = 500

print("Loading pkls...")
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

print("Rolling out trajectories...")
rng = random.PRNGKey(0)

def one_rollout(rollout_rng):
  states, _, _ = mdp.rollout(rollout_rng, config.env,
                             run_ddpg.policy(actor_params),
                             config.episode_length)
  return states

support_set = jit(vmap(one_rollout))(random.split(rng, num_rollouts))
support_set_flat = jp.reshape(support_set, (-1, support_set.shape[-1]))

plt.figure()
plt.scatter(support_set_flat[:, 0], support_set_flat[:, 1], alpha=0.1)
plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title(f"Seed {best_seed}")
plt.show()
