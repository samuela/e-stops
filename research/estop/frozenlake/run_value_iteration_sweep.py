import matplotlib
import matplotlib.pyplot as plt
import tqdm
import numpy as np

from research.estop.frozenlake import frozenlake

def build_env(lake: frozenlake.Lake):
  # return frozenlake.FrozenLakeEnv(lake, infinite_time=False)
  return frozenlake.FrozenLakeWithEscapingEnv(lake,
                                              hole_retention_probability=0.99)

def main():
  np.random.seed(0)

  lake_map = frozenlake.MAP_8x8
  gamma = 0.99

  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)

  state_action_values, _ = frozenlake.value_iteration(env,
                                                      gamma,
                                                      tolerance=1e-6)
  policy_actions = np.argmax(state_action_values, axis=-1)

  policy_transitions = np.array([
      env.transitions[i, policy_actions[i], :] for i in range(lake.num_states)
  ])
  hp, _ = frozenlake.markov_chain_stats(env, policy_transitions)

  # # Ensure that we don't remove the start state!
  # hp[lake.start_state] = 1.0

  # # Show hitting probability map.
  # plt.figure()
  # viz.plot_heatmap(lake, hp)
  # plt.title("Hitting probabilities")
  # plt.show()

  # Estimated hitting probabilities
  # estimated_hp = frozenlake.estimate_hitting_probabilities(
  #     env,
  #     frozenlake.deterministic_policy(env, policy_actions),
  #     num_rollouts=1000)

  # Show hitting probability map.
  # plt.figure()
  # viz.plot_heatmap(lake, estimated_hp)
  # plt.title("Estimated hitting probabilities")
  # plt.show()

  # See https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy.
  rank_hp2d = lake.reshape(np.argsort(np.argsort(hp)))

  def run(num_states_to_remove: int):
    estop_map = np.copy(lake_map)
    estop_map[rank_hp2d < num_states_to_remove] = "E"
    # print(num_states_to_remove)
    # print(estop_map)

    # Check that we haven't gotten rid of the start state yet.
    if (estop_map == "S").sum() == 0:
      return None

    estop_env = build_env(frozenlake.Lake(estop_map))
    _, policy_rewards_per_iter = frozenlake.value_iteration(
        estop_env,
        gamma,
        max_iterations=5000,
    )

    # plt.figure()
    # viz.plot_heatmap(frozenlake.Lake(estop_map), np.max(estop_state_action_values, axis=-1))
    # plt.title(f"V(s) with {num_states_to_remove} states removed")
    # plt.show()

    num_states = frozenlake.num_mdp_states(estop_map)
    # There are 4 S * A * S FLOPS in each iteration:
    #   * multiplying transitions with state_values
    #   * multiplying times gamma
    #   * adding expected_rewards
    #   * max'ing over state_action_values
    flops_per_iter = 4 * (frozenlake.NUM_ACTIONS *
                          (num_states**2)) * np.arange(
                              len(policy_rewards_per_iter))
    return flops_per_iter, policy_rewards_per_iter

  results = [run(i) for i in tqdm.trange(lake.num_states)]

  # Some of the maps don't have a feasible path to the goal so they're just zero
  # the whole time.
  noncrappy_results = [
      i for i, res in enumerate(results) if res is not None and res[1][-1] > 0
  ]

  plt.rcParams.update({"font.size": 16})
  cmap = plt.get_cmap("YlOrRd")

  plt.figure()
  for i, ix in enumerate(noncrappy_results):
    plt.plot(results[ix][0] / 1000.0,
             results[ix][1],
             color=cmap(i / len(noncrappy_results)))

  plt.xlim(0, 5e3)
  plt.xlabel("FLOPs (thousands)")
  plt.ylabel("Cumulative policy reward")
  colorbar = plt.colorbar(
      matplotlib.cm.ScalarMappable(
          cmap=cmap,
          norm=matplotlib.colors.Normalize(vmin=0,
                                           vmax=100 * max(noncrappy_results) /
                                           lake.num_states)))
  colorbar.set_label("E-stop states (%)", rotation=270, labelpad=25)
  plt.tight_layout()
  plt.savefig("figs/value_iteration_sweep.pdf")

  plt.figure()
  plt.plot(100 * np.array(noncrappy_results) / lake.num_states,
           [results[i][1][-1] for i in noncrappy_results])
  plt.xlabel("States removed (%)")
  plt.ylabel("Optimal policy cumulative reward")
  plt.tight_layout()
  plt.savefig("figs/num_removed_vs_policy_reward.pdf")

if __name__ == "__main__":
  main()
