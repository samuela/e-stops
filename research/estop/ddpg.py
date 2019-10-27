from functools import partial
from typing import Any, NamedTuple, Callable, Generic

from jax import jit, lax, grad, random, tree_util, vmap
import jax.numpy as jp

from research.estop.mdp import Env, State
from research.estop.replay_buffers import ReplayBuffer
from research.statistax import Distribution
from research.utils import Optimizer

@partial(jit, static_argnums=(3, ))
def noisy_actor(rng, actor_params, state, actor, noise):
  actor_action = actor(actor_params, state)
  action_noise = noise.sample(rng)

  # We corrupt the actor_action with noise in order to promote exploration.
  action = actor_action + action_noise
  return action, action_noise

def step_and_update_replay_buffer(
    rng,
    params,
    env: Env,
    replay_buffer: ReplayBuffer,
    actor,
    state,
    noise: Distribution,
    terminal_criterion,
):
  """Step the environment and add the transition to the replay buffer.

  Args:
    rng: The PRNG key.
    params (tuple): The current (actor, critic) parameters.
    env (Env): The environment to operate in.
    replay_buffer (ReplayBuffer): The experience replay buffer.
    actor: The actor function, `\\mu(s)`.
    state: The current state.
    noise: A function for the action noise distribution at each time step.
    terminal_criterion
  """
  actor_params, _ = params
  rng_noise, rng_transition = random.split(rng)
  action, action_noise = noisy_actor(rng_noise, actor_params, state, actor,
                                     noise)
  next_state = env.step(state, action).sample(rng_transition)
  reward = env.reward(state, action, next_state)
  done = terminal_criterion(next_state)
  new_rb = replay_buffer.add(state, action, reward, next_state, done)

  return action_noise, reward, next_state, done, new_rb

def ddpg_gradients(
    params,
    tracking_params,
    replay_minibatch,
    gamma: float,
    actor,
    critic,
):
  """Calculate gradients on the actor and critic based on a sample from the
  replay buffer.

  Args:
    params (tuple): The current (actor, critic) parameters.
    tracking_params (tuple): The tracking (actor, critic) parameters.
    replay_minibatch
    gamma (float): The time discount factor.
    actor: The actor function, `\\mu(s)`.
    critic: The critic function, `Q(s, a)`.
  """
  actor_params, critic_params = params
  tracking_actor_params, tracking_critic_params = tracking_params
  Q_track = lambda s, a: critic(tracking_critic_params, (s, a))
  mu_track = lambda s: actor(tracking_actor_params, s)
  replay_states, replay_actions, replay_rewards, replay_next_states, replay_done = replay_minibatch

  replay_ys = vmap(lambda r, ns, d: r + gamma * lax.bitwise_not(d) * Q_track(
      ns, mu_track(ns)),
                   in_axes=(0, 0, 0))(
                       replay_rewards,
                       replay_next_states,
                       replay_done,
                   )

  def critic_loss(p):
    replay_pred_ys = vmap(lambda s, a: critic(p, (s, a)),
                          in_axes=(0, 0))(replay_states, replay_actions)
    return jp.mean((replay_ys - replay_pred_ys)**2.0)

  critic_grad = grad(critic_loss)(critic_params)

  def actor_loss(p):
    # Easier to represent it this way instead of expanding the chain rule, as is
    # done in the paper.
    loss_single = lambda s: -critic(critic_params, (s, actor(p, s)))
    return jp.mean(vmap(loss_single)(replay_states))

  actor_grad = grad(actor_loss)(actor_params)

  # Note: There's potentially a slight deviation from the paper here in the
  # sense that it says "update the critic, then update the actor" but the
  # gradient of the actor depends on the critic. For simplicity, this
  # implementation calculates the gradient of the actor without updating the
  # critic first. This should have a negligible impact on behavior.
  return (actor_grad, critic_grad)

@partial(jit, static_argnums=(3, 4, 5, 6))
def ddpg_update(
    params_optimizer,
    tracking_params,
    replay_minibatch,
    gamma: float,
    tau: float,
    actor,
    critic,
):
  g = ddpg_gradients(
      params_optimizer.value,
      tracking_params,
      replay_minibatch,
      gamma,
      actor,
      critic,
  )
  new_optimizer = params_optimizer.update(g)
  new_tracking_params = tree_util.tree_multimap(
      lambda new, old: tau * new + (1 - tau) * old,
      new_optimizer.value,
      tracking_params,
  )
  return new_optimizer, new_tracking_params

class DDPGStepOut(Generic[State], NamedTuple):
  action_noise: jp.ndarray
  reward: jp.ndarray
  next_state: State
  done: jp.ndarray
  replay_buffer: ReplayBuffer
  optimizer: Optimizer
  tracking_params: Any

def ddpg_step(
    rng,
    params_optimizer: Optimizer,
    tracking_params,
    env: Env,
    gamma: float,
    tau: float,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    actor,
    critic,
    state,
    noise: Distribution,
    terminal_criterion,
) -> DDPGStepOut:
  """Step the environment, update replay buffer, calculate gradients, and update
  the parameters.

  Args:
    rng: The PRNG key.
    params (tuple): The current (actor, critic) parameters.
    tracking_params (tuple): The tracking (actor, critic) parameters.
    env (Env): The environment to operate in.
    gamma (float): The time discount factor.
    tau (float)
    replay_buffer (ReplayBuffer): The experience replay buffer.
    batch_size (int): The size of the batch to pull out of `replay_buffer`.
    actor: The actor function, `\\mu(s)`.
    critic: The critic function, `Q(s, a)`.
    state: The current state.
    noise: A function for the action noise distribution at each time step.
    terminal_criterion
  """
  rng_step, rng_minibatch = random.split(rng)

  action_noise, reward, next_state, done, new_rb = step_and_update_replay_buffer(
      rng_step,
      params_optimizer.value,
      env,
      replay_buffer,
      actor,
      state,
      noise,
      terminal_criterion,
  )

  # Sample minibatch from the replay buffer.
  replay_minibatch = new_rb.minibatch(rng_minibatch, batch_size)

  new_optimizer, new_tracking_params = ddpg_update(
      params_optimizer,
      tracking_params,
      replay_minibatch,
      gamma,
      tau,
      actor,
      critic,
  )

  return DDPGStepOut(action_noise, reward, next_state, done, new_rb,
                     new_optimizer, new_tracking_params)

class LoopState(Generic[State], NamedTuple):
  episode_length: int
  optimizer: Optimizer
  tracking_params: Any
  discounted_cumulative_reward: jp.ndarray
  undiscounted_cumulative_reward: jp.ndarray
  state: State
  replay_buffer: ReplayBuffer
  prev_noise: jp.ndarray

  # This is actually a jax type because it has to be traced. Should just be a
  # scalar with dtype=bool.
  done: jp.ndarray

def ddpg_episode(
    env: Env,
    gamma: float,
    tau: float,
    actor,
    critic,
    noise: Callable[[int, jp.ndarray], Distribution],
    terminal_criterion,
    batch_size: int,
    while_loop=lax.while_loop,
):
  """Run DDPG for a single episode.

  Args:
    env (Env): The environment to run the agent in.
    gamma (float): The time discount factor.
    tau (float): The parameter tracking rate.
    actor: The actor function, `\\mu(s)`.
    critic: The critic function, `Q(s, a)`.
    noise: A function for the action noise distribution at each time step.
    episode_length (int): The length of the episode to run.
    batch_size (int): The size of batches to be pulled out of the replay buffer.

  Returns:
    run: A `jit`-able function to actually run the episode.
  """
  def run(
      rng,
      init_noise: Distribution,
      init_replay_buffer: ReplayBuffer,
      init_optimizer: Optimizer,
      init_tracking_params,
  ) -> LoopState:
    """A curried `jit`-able function to actually run the DDPG episode."""
    rng_init_state, rng_init_noise, rng_steps = random.split(rng, 3)

    def step(loop_state: LoopState):
      t = loop_state.episode_length
      step_out = ddpg_step(
          random.fold_in(rng_steps, t),
          loop_state.optimizer,
          loop_state.tracking_params,
          env,
          gamma,
          tau,
          loop_state.replay_buffer,
          batch_size,
          actor,
          critic,
          loop_state.state,
          noise(t, loop_state.prev_noise),
          lambda s: terminal_criterion(t, s),
      )

      new_discounted_cumulative_reward = loop_state.discounted_cumulative_reward + (
          gamma**t) * step_out.reward
      new_undiscounted_cumulative_reward = (
          loop_state.undiscounted_cumulative_reward + step_out.reward)
      return LoopState(
          episode_length=loop_state.episode_length + 1,
          optimizer=step_out.optimizer,
          tracking_params=step_out.tracking_params,
          discounted_cumulative_reward=new_discounted_cumulative_reward,
          undiscounted_cumulative_reward=new_undiscounted_cumulative_reward,
          state=step_out.next_state,
          replay_buffer=step_out.replay_buffer,
          prev_noise=step_out.action_noise,
          done=step_out.done,
      )

    init_val = LoopState(
        episode_length=0,
        optimizer=init_optimizer,
        tracking_params=init_tracking_params,
        discounted_cumulative_reward=jp.array(0.0),
        undiscounted_cumulative_reward=jp.array(0.0),
        state=env.initial_distribution.sample(rng_init_state),
        replay_buffer=init_replay_buffer,
        prev_noise=init_noise.sample(rng_init_noise),
        done=jp.array(False),
    )

    return while_loop(lambda ls: lax.bitwise_not(ls.done), step, init_val)

  return run
