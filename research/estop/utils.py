import jax.numpy as jp

from research.statistax import Normal

def Scalarify():
  def init_fn(_rng, input_shape):
    assert input_shape[-1] == 1
    return input_shape[:-1], ()

  def apply_fn(_, inputs, **_2):
    return inputs[..., 0]

  return init_fn, apply_fn

Scalarify = Scalarify()

def ornstein_uhlenbeck_noise(mu: float,
                             sigma: float = 0.1,
                             theta: float = 0.15,
                             dt: float = 1e-2):
  scale = sigma * jp.sqrt(dt)

  def step(_, x_prev):
    return Normal(x_prev + theta * (mu - x_prev) * dt, scale)

  return step
