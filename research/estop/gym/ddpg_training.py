"""Utilities for running DDPG on gym environments, esp. mujoco."""

from pathlib import Path
from typing import Any, NamedTuple, Optional
import time
import datetime
import pickle

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from jax import jit, random
from jax.experimental import optimizers
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental.stax import FanInConcat, Dense, Relu, Tanh

from research import flax
from research.estop import replay_buffers, ddpg
from research.estop.gym.gym_wrappers import GymEnvSpec, build_env_spec
from research.estop.utils import Scalarify
from research.statistax import Deterministic, Normal
from research.utils import make_optimizer

class DDPGTrainConfig(NamedTuple):
  gamma: float
  tau: float
  buffer_size: int
  batch_size: int
  optimizer_init: Any
  noise: Any

  actor_init: Any
  actor: Any
  critic_init: Any
  critic: Any

def deterministic_policy(train_config: DDPGTrainConfig, actor_params):
  return jit(lambda s: Deterministic(train_config.actor(actor_params, s)))

def make_default_ddpg_train_config(env_spec: GymEnvSpec):
  """Usually decent parameters."""
  # Actions must be bounded [-1, 1].
  actor_init, actor = stax.serial(
      Dense(64),
      Relu,
      Dense(64),
      Relu,
      Dense(env_spec.action_shape[0]),
      Tanh,
  )

  critic_init, critic = stax.serial(
      FanInConcat(),
      Dense(64),
      Relu,
      Dense(64),
      Relu,
      Dense(64),
      Relu,
      Dense(1),
      Scalarify,
  )
  return DDPGTrainConfig(
      gamma=0.99,
      tau=1e-4,
      buffer_size=2**20,
      batch_size=128,
      optimizer_init=make_optimizer(optimizers.adam(step_size=1e-3)),
      # For some reason using DiagMVN here is ~100x slower.
      noise=lambda _1, _2: Normal(
          jp.zeros(env_spec.action_shape),
          0.1 * jp.ones(env_spec.action_shape),
      ),
      actor_init=actor_init,
      actor=actor,
      critic_init=critic_init,
      critic=critic,
  )

def rollout(env_spec: GymEnvSpec, rng, policy, callback=lambda: None):
  """Rollout one episode of the half_cheetah environment. This is specialized to
  the fact that we can't use lax.scan and the policy and environment dynamics
  are deterministic."""
  init_state = env_spec.env.initial_distribution.sample(rng)

  states = [init_state]
  actions = []
  rewards = []
  for _ in range(env_spec.max_episode_steps):
    state = states[-1]

    # These are specialized to the fact the policy and the environment are both
    # deterministic because it's ~10x faster.
    action = policy(state).loc
    next_state = env_spec.env.step(state, action).loc
    reward = env_spec.env.reward(state, action, next_state)

    states.append(next_state)
    actions.append(action)
    rewards.append(reward)

    # Here we let the caller render or capture a frame for video.
    callback()

  # Drop the last state so that all of the lists have the same length.
  states = states[:-1]

  return states, actions, rewards

def eval_policy(env_spec: GymEnvSpec, rng, policy, num_eval_rollouts: int):
  """Run policy evaluation by simply rolling out a bunch of trajectories."""
  total_reward = 0.0
  for r in random.split(rng, num_eval_rollouts):
    _, _, rewards = rollout(env_spec, r, policy)
    total_reward += np.sum(rewards)
  return total_reward / num_eval_rollouts

def film_policy(env_spec: GymEnvSpec, rng, policy, filepath: Path):
  """Create a video of the policy running."""
  video_env = VideoRecorder(env_spec.gym_env, path=str(filepath))
  # pylint: disable=unnecessary-lambda
  rollout(env_spec, rng, policy, callback=lambda: video_env.capture_frame())
  video_env.close()

def train(
    train_config: DDPGTrainConfig,
    env_spec: GymEnvSpec,
    rng,
    num_episodes,
    terminal_criterion,
    callback,
):
  """Generic DDPG training loop that offers a callback for extensibility."""
  actor_init_rng, critic_init_rng, rng = random.split(rng, 3)
  _, init_actor_params = train_config.actor_init(actor_init_rng,
                                                 env_spec.state_shape)
  _, init_critic_params = train_config.critic_init(
      critic_init_rng, (env_spec.state_shape, env_spec.action_shape))
  optimizer = train_config.optimizer_init(
      (init_actor_params, init_critic_params))
  tracking_params = optimizer.value

  replay_buffer = replay_buffers.NumpyReplayBuffer(train_config.buffer_size,
                                                   env_spec.state_shape,
                                                   env_spec.action_shape)

  run = ddpg.ddpg_episode(
      env=env_spec.env,
      gamma=train_config.gamma,
      tau=train_config.tau,
      actor=train_config.actor,
      critic=train_config.critic,
      noise=train_config.noise,
      terminal_criterion=terminal_criterion,
      batch_size=train_config.batch_size,
      # We need a flax loop here since we can't jit compile mujoco steps.
      while_loop=flax.while_loop,
  )

  episode_rngs = random.split(rng, num_episodes)
  init_noise = Deterministic(jp.zeros(env_spec.action_shape))
  for episode in range(num_episodes):
    t0 = time.time()
    final_state = run(
        episode_rngs[episode],
        init_noise,
        replay_buffer,
        optimizer,
        tracking_params,
    )
    if not jp.isfinite(final_state.undiscounted_cumulative_reward):
      raise Exception("Reached non-finite reward. Probably a NaN.")

    callback({
        "episode": episode,
        "episode_length": final_state.episode_length,
        "optimizer": final_state.optimizer,
        "tracking_params": final_state.tracking_params,
        "discounted_cumulative_reward":
        final_state.discounted_cumulative_reward,
        "undiscounted_cumulative_reward":
        final_state.undiscounted_cumulative_reward,
        "elapsed": time.time() - t0,
    })

    # Update loop variables...
    optimizer = final_state.optimizer
    tracking_params = final_state.tracking_params
    replay_buffer = final_state.replay_buffer

def debug_run(
    env_spec: GymEnvSpec,
    train_config: DDPGTrainConfig,
    seed: int = 0,
    num_episodes: int = 10000,
    state_min: Optional[np.ndarray] = None,
    state_max: Optional[np.ndarray] = None,
    num_eval_rollouts: int = 8,
    policy_evaluation_frequency: int = 1000,
    policy_video_frequency: int = 1000,
    respect_gym_done: bool = False,
):
  """A debug training loop designed to be used for local testing."""
  rng = random.PRNGKey(seed)
  np.random.seed(seed)

  train_rng, rng = random.split(rng)
  callback_rngs = random.split(rng, num_episodes)

  results_dir = Path(
      f"results/debug_ddpg_{env_spec.env_name}_{datetime.datetime.utcnow()}/")
  results_dir.mkdir()

  def callback(info):
    episode = info["episode"]
    episode_length = info["episode_length"]
    discounted_cumulative_reward = info["discounted_cumulative_reward"]
    undiscounted_cumulative_reward = info["undiscounted_cumulative_reward"]
    current_actor_params, _ = info["optimizer"].value

    print(f"Episode {episode}")
    print("  disc. reward = {0:.4f}".format(discounted_cumulative_reward))
    print("  undisc. reward = {0:.4f}".format(undiscounted_cumulative_reward))
    print("  ep. length = {}".format(episode_length))
    print("  reward/step = {0:.4f}".format(undiscounted_cumulative_reward /
                                           episode_length))
    print("  elapsed = {0:.4f}".format(info["elapsed"]))

    # Periodically evaluate the policy without any action noise.
    if episode % policy_evaluation_frequency == 0:
      curr_policy = deterministic_policy(train_config, current_actor_params)
      tic = time.time()
      policy_value = eval_policy(env_spec, callback_rngs[episode], curr_policy,
                                 num_eval_rollouts)
      print(
          f".. policy value (undisc.) = {policy_value}, elapsed = {time.time() - tic}"
      )

    if (episode + 1) % policy_video_frequency == 0:
      curr_policy = deterministic_policy(train_config, current_actor_params)
      tic = time.time()
      film_policy(env_spec,
                  callback_rngs[episode],
                  curr_policy,
                  filepath=results_dir / f"episode_{episode}.mp4")
      print(f".. saved episode video, elapsed = {time.time() - tic}")

  state_min = state_min if state_min is not None else -np.inf * np.ones(
      env_spec.state_shape)
  state_max = state_max if state_max is not None else np.inf * np.ones(
      env_spec.state_shape)

  train(
      train_config=train_config,
      env_spec=env_spec,
      rng=train_rng,
      num_episodes=num_episodes,
      terminal_criterion=lambda t, s:
      ((t >= env_spec.max_episode_steps) or np.any(s < state_min) or np.any(
          s > state_max) or (respect_gym_done and env_spec.gym_env.done)),
      callback=callback,
  )

def batch_job(
    random_seed: int,
    env_name: str,
    reward_adjustment: float,
    num_episodes: int,
    state_min,
    state_max,
    out_dir: Path,
    num_eval_rollouts: int = 64,
    policy_evaluation_frequency: int = 100,
    policy_video_frequency: int = 1000,
    respect_gym_done: bool = False,
):
  """Run a "batch" training job.

  Note that only pickle-able things may be arguments to this function so that we
  can use it with multiprocessing. This is why we need to pass in `env_name` and
  `reward_adjustment`, for example."""

  job_dir = out_dir / f"seed={random_seed}"
  job_dir.mkdir()

  rng = random.PRNGKey(random_seed)
  np.random.seed(random_seed)

  callback_rng, train_rng = random.split(rng)
  callback_rngs = random.split(callback_rng, num_episodes)

  env_spec = build_env_spec(env_name, reward_adjustment)
  train_config = make_default_ddpg_train_config(env_spec)

  params = [None]
  tracking_params = [None]
  discounted_cumulative_reward_per_episode = []
  undiscounted_cumulative_reward_per_episode = []
  policy_evaluations = []
  episode_lengths = []
  elapsed_per_episode = []

  def callback(info):
    episode = info['episode']
    params[0] = info["optimizer"].value
    tracking_params[0] = info["tracking_params"]

    current_actor_params, _ = info["optimizer"].value

    discounted_cumulative_reward_per_episode.append(
        info["discounted_cumulative_reward"])
    undiscounted_cumulative_reward_per_episode.append(
        info["undiscounted_cumulative_reward"])
    episode_lengths.append(info["episode_length"])
    elapsed_per_episode.append(info["elapsed"])

    if episode % policy_evaluation_frequency == 0:
      curr_policy = deterministic_policy(train_config, current_actor_params)
      policy_value = eval_policy(env_spec, callback_rngs[episode], curr_policy,
                                 num_eval_rollouts)
      policy_evaluations.append(policy_value)

    if (episode + 1) % policy_video_frequency == 0:
      curr_policy = deterministic_policy(train_config, current_actor_params)
      film_policy(env_spec,
                  callback_rngs[episode],
                  curr_policy,
                  filepath=job_dir / f"episode_{episode}.mp4")

  train(
      train_config,
      env_spec,
      train_rng,
      num_episodes,
      lambda t, s:
      ((t >= env_spec.max_episode_steps) or np.any(s < state_min) or np.any(
          s > state_max) or (respect_gym_done and env_spec.gym_env.done)),
      callback,
  )
  with (job_dir / f"data.pkl").open(mode="wb") as f:
    pickle.dump(
        {
            "env_name": env_name,
            "reward_adjustment": reward_adjustment,
            "final_params": params[0],
            "final_tracking_params": tracking_params[0],
            "discounted_cumulative_reward_per_episode":
            discounted_cumulative_reward_per_episode,
            "undiscounted_cumulative_reward_per_episode":
            undiscounted_cumulative_reward_per_episode,
            "policy_evaluations": policy_evaluations,
            "episode_lengths": episode_lengths,
            "elapsed_per_episode": elapsed_per_episode,
            "state_min": state_min,
            "state_max": state_max,
        }, f)
