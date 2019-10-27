from jax.experimental import optimizers
import numpy as np

class JaxAdam:
  def __init__(self, x0, learning_rate):
    self.opt_init, self.opt_update, self.get_params = optimizers.adam(
        step_size=learning_rate)

    self.opt_state = self.opt_init(x0)
    self.iteration = 0

  def step(self, gradient):
    self.opt_state = self.opt_update(self.iteration, gradient, self.opt_state)
    self.iteration += 1

  def get(self):
    return self.get_params(self.opt_state)

class Adam:
  def __init__(self, x0, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    self.x = x0
    self.learning_rate = learning_rate
    self.b1 = b1
    self.b2 = b2
    self.eps = eps

    self.m = np.zeros_like(x0)
    self.v = np.zeros_like(x0)
    self.iteration = 0

  def step(self, gradient):
    # self.m = (1 - self.b1) * gradient + self.b1 * self.m
    self.m *= self.b1
    self.m += (1 - self.b1) * gradient

    # self.v = (1 - self.b2) * (gradient**2) + self.b2 * self.v
    self.v *= self.b2
    self.v += (1 - self.b2) * (gradient**2)

    mhat = self.m / (1 - self.b1**(self.iteration + 1))
    vhat = self.v / (1 - self.b2**(self.iteration + 1))
    self.x -= self.learning_rate * mhat / (np.sqrt(vhat) + self.eps)
    self.iteration += 1

  def get(self):
    return self.x

class Momentum:
  def __init__(self, x0, learning_rate, mass=0.9):
    self.x = x0
    self.learning_rate = learning_rate
    self.mass = mass

    self.velocity = np.zeros_like(x0)

  def step(self, gradient):
    self.velocity = self.mass * self.velocity - (1.0 - self.mass) * gradient
    self.x += self.learning_rate * self.velocity

  def get(self):
    return self.x
