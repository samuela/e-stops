import matplotlib.pyplot as plt
import tqdm
import numpy as np

from research.estop.frozenlake import frozenlake

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
  num_random_policies = 1024

  Q, _ = frozenlake.value_iteration(env, gamma, tolerance=1e-6)

  def estop_map_optimal_policy_value(hp):
    # See https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy.
    rank_hp2d = lake.reshape(np.argsort(np.argsort(hp)))

    estop_map = np.copy(lake_map)
    estop_map[rank_hp2d < num_states_to_remove] = "E"

    # Check that we haven't gotten rid of the start state yet.
    if (estop_map == "S").sum() == 0:
      # This could also be recorded as zero, depending on how you want to think
      # about it.
      return None

    estop_env = build_env(frozenlake.Lake(estop_map))
    return frozenlake.optimal_policy_reward(estop_env, gamma)

  def one(noise_scale: float):
    policy_actions = np.argmax(Q + noise_scale * np.random.randn(*Q.shape),
                               axis=-1)
    policy = frozenlake.deterministic_policy(env, policy_actions)
    V, _ = frozenlake.iterative_policy_evaluation(env,
                                                  gamma,
                                                  policy,
                                                  tolerance=1e-6)
    policy_value = np.dot(V, env.initial_state_distribution)

    # Calculate the value of the optimal policy in the exact e-stop environment.
    policy_transitions = np.array([
        env.transitions[i, policy_actions[i], :]
        for i in range(lake.num_states)
    ])
    try:
      exact_hp, _ = frozenlake.markov_chain_stats(env, policy_transitions)
    except np.linalg.LinAlgError:
      # Sometimes the policy is bad and one of the matrices ends up singular.
      return None

    estop_policy_value = estop_map_optimal_policy_value(exact_hp)
    if estop_policy_value is None:
      return None

    return policy_value, estop_policy_value

  results = np.array([
      x for x in [
          one(s)
          for s in tqdm.tqdm(np.linspace(0, 0.1, num=num_random_policies))
      ] if x is not None
  ])

  plt.rcParams.update({"font.size": 16})
  plt.figure()
  plt.scatter(results[:, 0], results[:, 1], alpha=0.5)
  plt.plot([0, np.max(results)], [0, np.max(results)],
           color="grey",
           linestyle="--")
  plt.xlabel("Expert policy cumulative reward")
  plt.ylabel("E-stop policy cumulative reward")
  plt.tight_layout()
  plt.savefig("figs/expert_quality_vs_estop_perf.pdf")
