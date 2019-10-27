import matplotlib.pyplot as plt
import numpy as np

from research.estop.frozenlake import frozenlake
from research.estop.frozenlake import viz

def build_env(lake: frozenlake.Lake):
  # return frozenlake.FrozenLakeEnv(lake, infinite_time=False)
  return frozenlake.FrozenLakeWithEscapingEnv(lake,
                                              hole_retention_probability=0.99)

def main():
  # pylint: disable=too-many-statements
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  gamma = 0.99

  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  state_action_values, policy_rewards_per_iter = frozenlake.value_iteration(
      env, gamma, tolerance=1e-6)
  policy_actions = np.argmax(state_action_values, axis=-1)
  state_values = np.max(state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  viz.plot_heatmap(lake, state_values)
  # plt.title("FrozenLake-v0 environment")
  plt.tick_params(
      axis="both",
      which="both",
      bottom=False,
      top=False,
      left=False,
      right=False,
      labelbottom=False,
      labeltop=False,
      labelleft=False,
      labelright=False,
  )
  plt.tight_layout()
  plt.savefig("figs/value_function_full_env.pdf")

  # Show hitting probability map.
  policy_transitions = np.array([
      env.transitions[i, policy_actions[i], :] for i in range(lake.num_states)
  ])
  hp, esta = frozenlake.markov_chain_stats(env, policy_transitions)
  hp2d = lake.reshape(hp)

  plt.figure()
  viz.plot_heatmap(lake, hp)
  plt.title("Hitting probabilities")
  plt.savefig("figs/hitting_probabilities.pdf")

  # Show estimated hitting probability map.
  estimated_hp = frozenlake.estimate_hitting_probabilities(
      env,
      frozenlake.deterministic_policy(env, policy_actions),
      num_rollouts=1000)
  plt.figure()
  viz.plot_heatmap(lake, estimated_hp)
  plt.title("Estimated hitting probabilities")

  plt.figure()
  viz.plot_heatmap(lake, esta)
  plt.title("Expected number of states to completion")

  # Show optimal policy on top of hitting probabilities.
  plt.figure()
  im = plt.imshow(hp2d)
  for s, a in zip(lake.ij_states, policy_actions):
    i, j = s
    if a == 0:
      arrow = "←"
    elif a == 1:
      arrow = "↓"
    elif a == 2:
      arrow = "→"
    elif a == 3:
      arrow = "↑"
    else:
      raise Exception("bad bad bad")

    im.axes.text(j, i, arrow, {
        "horizontalalignment": "center",
        "verticalalignment": "center"
    })
  plt.title("Optimal policy overlayed on hitting probabilities")
  plt.savefig("figs/optimal_policy.pdf")

  # Show value CDF.
  plt.figure()
  plt.hist(state_values, bins=100, histtype="step", cumulative=True)
  plt.xlabel("V(s)")
  plt.ylabel(f"Number of states (out of {lake.num_states})")
  plt.title("CDF of state values")
  plt.savefig("figs/value_function_cdf.pdf")

  #######

  # New map has hole everywhere with bad prob.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  # Use less than or equal because the estimated hitting probabilities can be
  # zero and the threshold can be zero, so nothing on the map changes.
  estop_map[lake.reshape(estimated_hp) <= threshold] = "E"

  estop_lake = frozenlake.Lake(estop_map)
  estop_env = build_env(estop_lake)
  estop_state_action_values, estop_policy_rewards_per_iter = frozenlake.value_iteration(
      estop_env, gamma, tolerance=1e-6)
  estop_state_values = np.max(estop_state_action_values, axis=-1)

  # Show value function map.
  plt.figure()
  viz.plot_heatmap(estop_lake, estop_state_values)
  plt.title(f"E-stop map ({percentile}% of states removed)")
  plt.savefig("figs/estop_map.pdf")

  # Show policy rewards per iter
  # There are 4 S * A * S FLOPS in each iteration:
  #   * multiplying transitions with state_values
  #   * multiplying times gamma
  #   * adding expected_rewards
  #   * max'ing over state_action_values

  plt.figure()
  plt.plot(
      4 * (frozenlake.NUM_ACTIONS * (frozenlake.num_mdp_states(lake_map)**2)) *
      np.arange(len(policy_rewards_per_iter)), policy_rewards_per_iter)
  plt.plot(
      4 * (frozenlake.NUM_ACTIONS *
           (frozenlake.num_mdp_states(estop_map)**2)) *
      np.arange(len(estop_policy_rewards_per_iter)),
      estop_policy_rewards_per_iter)
  plt.xlabel("FLOPS")
  plt.ylabel("Policy reward")
  plt.legend(["Full MDP", "E-stop MDP"])
  plt.title("Convergence comparison")
  plt.savefig("figs/convergence_comparison.pdf")

  print(
      f"Exact solution, policy value: {np.dot(env.initial_state_distribution, state_values)}"
  )
  print(
      f"E-stop solution, policy value: {np.dot(env.initial_state_distribution, estop_state_values)}"
  )

  plt.show()

if __name__ == "__main__":
  main()
