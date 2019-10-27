import matplotlib.pyplot as plt
import numpy as np

from research.estop.frozenlake import frozenlake
from research.estop.frozenlake import optimizers
from research.estop.frozenlake import reinforce
from research.estop.frozenlake import viz

def build_env(lake: frozenlake.Lake):
  # return frozenlake.FrozenLakeEnv(lake, infinite_time=True)
  return frozenlake.FrozenLakeWithEscapingEnv(lake,
                                              hole_retention_probability=0.99)

def main():
  np.random.seed(0)

  # lake_map = frozenlake.MAP_CORRIDOR_4x1
  lake_map = frozenlake.MAP_8x8
  policy_evaluation_frequency = 100
  gamma = 0.99

  lake = frozenlake.Lake(lake_map)
  env = build_env(lake)
  print(
      f"Optimal policy reward on full env: {frozenlake.optimal_policy_reward(env, gamma)}"
  )

  # Estimate hitting probabilities.
  state_action_values, _ = frozenlake.value_iteration(
      env,
      gamma,
      tolerance=1e-6,
  )
  optimal_policy = frozenlake.deterministic_policy(
      env, np.argmax(state_action_values, axis=-1))
  estimated_hp = frozenlake.estimate_hitting_probabilities(
      env,
      optimal_policy,
      num_rollouts=1000,
  )
  estimated_hp2d = lake.reshape(estimated_hp)

  # Build e-stop environment.
  estop_map = np.copy(lake_map)
  percentile = 50
  threshold = np.percentile(estimated_hp, percentile)
  estop_map[estimated_hp2d <= threshold] = "E"

  estop_lake = frozenlake.Lake(estop_map)
  estop_env = build_env(estop_lake)
  print(
      f"Optimal policy reward on e-stop: {frozenlake.optimal_policy_reward(estop_env, gamma)}"
  )

  plt.figure()
  viz.plot_heatmap(estop_lake, np.zeros(estop_lake.num_states))
  plt.title("E-stop map")

  plt.figure()
  viz.plot_heatmap(lake, np.zeros(lake.num_states))
  plt.title("Full map")

  plt.show()

  plt.figure()
  for seed in range(1):
    np.random.seed(seed)

    x0 = 1e-2 * np.random.randn(estop_env.lake.num_states,
                                frozenlake.NUM_ACTIONS)
    optimizer = optimizers.Adam(x0, learning_rate=1e-3)
    # optimizer = reinforce.Momentum(x0, learning_rate=1e-2, mass=0.0)
    states_seen, policy_rewards = reinforce.run_reinforce(
        estop_env,
        gamma,
        optimizer,
        num_episodes=50000,
        policy_evaluation_frequency=policy_evaluation_frequency)

    plt.plot(states_seen, policy_rewards)

  plt.axhline(frozenlake.optimal_policy_reward(env, gamma),
              color="grey",
              linestyle="--")
  plt.axhline(frozenlake.optimal_policy_reward(estop_env, gamma),
              color="grey",
              linestyle="--")
  plt.title(f"Learning rate={optimizer.learning_rate}")
  plt.show()

if __name__ == "__main__":
  main()
