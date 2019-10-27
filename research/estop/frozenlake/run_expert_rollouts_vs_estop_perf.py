import matplotlib.pyplot as plt
import tqdm
import numpy as np

from research.estop.frozenlake import frozenlake, viz

nums_of_rollouts = range(1, 10)

def build_env(l: frozenlake.Lake):
  return frozenlake.FrozenLakeWithEscapingEnv(l,
                                              hole_retention_probability=0.99)

if __name__ == "__main__":
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  gamma = 0.99

  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  num_states_to_remove = 0.5 * lake.num_states

  def estop_map_optimal_policy_value(hp):
    # See https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy.
    rank_hp2d = lake.reshape(np.argsort(np.argsort(hp)))

    estop_map = np.copy(lake_map)
    estop_map[rank_hp2d < num_states_to_remove] = "E"

    # Check that we haven't gotten rid of the start state yet.
    if (estop_map == "S").sum() == 0:
      # This could also just be recorded as zero, depending on how you want to
      # think about it.
      return None

    estop_env = build_env(frozenlake.Lake(estop_map))
    return frozenlake.optimal_policy_reward(estop_env, gamma)

  state_action_values, optimal_policy_values = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)

  # The value of the optimal policy in the full environment.
  opt_full_policy_value = optimal_policy_values[-1]
  policy_actions = np.argmax(state_action_values, axis=-1)

  # Calculate the value of the optimal policy in the exact e-stop environment.
  policy_transitions = np.array([
      env.transitions[i, policy_actions[i], :] for i in range(lake.num_states)
  ])
  exact_hp, _ = frozenlake.markov_chain_stats(env, policy_transitions)
  opt_estop_policy_value = estop_map_optimal_policy_value(exact_hp)

  def run(num_rollouts: int):
    policy_values = []
    while len(policy_values) < 64:
      # Estimated hitting probabilities
      estimated_hp = frozenlake.estimate_hitting_probabilities(
          env, frozenlake.deterministic_policy(env, policy_actions),
          num_rollouts)
      v = estop_map_optimal_policy_value(estimated_hp)
      if v is not None:
        policy_values.append(v)

    return policy_values

  results = np.array([run(i) for i in tqdm.tqdm(nums_of_rollouts)]).T

  plt.rcParams.update({"font.size": 16})

  plt.figure()
  viz.plot_errorfill(nums_of_rollouts, results, "blue")
  plt.axhline(opt_estop_policy_value,
              color="grey",
              linestyle="--",
              label="Optimal e-stop MDP")
  plt.xlabel("Expert trajectories observed")
  plt.ylabel("E-stop cumulative policy reward")
  plt.legend(loc="lower right")
  plt.tight_layout()
  plt.savefig("figs/expert_rollouts_vs_estop_perf.pdf")
