from typing import cast, Iterable, NamedTuple, Tuple, Callable, Any

import jax.numpy as jp
from jax import lax, random
from jax.scipy import linalg

NEG_HALF_LOG_TWO_PI = -0.5 * jp.log(2 * jp.pi)

class Distribution:
  @property
  def event_shape(self):
    raise NotImplementedError()

  @property
  def batch_shape(self):
    raise NotImplementedError()

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    raise NotImplementedError()

  def log_prob(self, x: jp.ndarray) -> jp.ndarray:
    raise NotImplementedError()

  def entropy(self) -> jp.ndarray:
    raise NotImplementedError()

class Normal(Distribution, NamedTuple):
  loc: jp.ndarray
  scale: jp.ndarray

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return lax.broadcast_shapes(self.loc.shape, self.scale.shape)

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    return self.loc + self.scale * random.normal(
        rng, shape=sample_shape + self.batch_shape)

  def log_prob(self, x) -> jp.ndarray:
    dists = 0.5 * ((x - self.loc) / self.scale)**2.0
    return -jp.log(self.scale) - dists + NEG_HALF_LOG_TWO_PI

  def entropy(self) -> jp.ndarray:
    return jp.log(self.scale) + 0.5 - NEG_HALF_LOG_TWO_PI

class Uniform(Distribution, NamedTuple):
  minval: jp.ndarray
  maxval: jp.ndarray

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return lax.broadcast_shapes(self.minval.shape, self.maxval.shape)

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    return self.minval + (self.maxval - self.minval) * random.uniform(
        rng, shape=sample_shape + self.batch_shape)

  def log_prob(self, x) -> jp.ndarray:
    return jp.where(
        self.minval <= x <= self.maxval,
        -jp.log(self.maxval - self.minval),
        jp.log(0),
    )

  def entropy(self) -> jp.ndarray:
    return jp.log(self.maxval - self.minval)

class Deterministic(Distribution, NamedTuple):
  loc: jp.ndarray
  eps: float = 0.0

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return self.loc.shape

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    return jp.broadcast_to(self.loc, shape=sample_shape + self.batch_shape)

  def log_prob(self, x) -> jp.ndarray:
    return jp.where(jp.abs(x - self.loc) <= self.eps, 0.0, jp.log(0))

  def entropy(self) -> jp.ndarray:
    return jp.zeros_like(self.loc)

class SampleOnly(Distribution, NamedTuple):
  # pylint: disable=abstract-method

  sample_fn: Callable[[Any], jp.ndarray]

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    # For now we don't bother supporting sample_shape.
    assert sample_shape == ()
    return self.sample_fn(rng)

def Independent(reinterpreted_batch_ndims: int):
  # pylint: disable=redefined-outer-name
  class Independent(Distribution, NamedTuple):
    base_distribution: Distribution

    @property
    def event_shape(self):
      return (self.base_distribution.batch_shape[-reinterpreted_batch_ndims:] +
              self.base_distribution.event_shape)

    @property
    def batch_shape(self):
      return self.base_distribution.batch_shape[:-reinterpreted_batch_ndims]

    def sample(self, rng, sample_shape=()) -> jp.ndarray:
      return self.base_distribution.sample(rng, sample_shape)

    def log_prob(self, x) -> jp.ndarray:
      # Will have shape [sample_shape, base.batch_shape].
      full = self.base_distribution.log_prob(x)
      return jp.sum(full, axis=tuple(range(-reinterpreted_batch_ndims, 0)))

    def entropy(self) -> jp.ndarray:
      # Will have shape base.batch_shape.
      full = self.base_distribution.entropy()
      return jp.sum(full, axis=tuple(range(-reinterpreted_batch_ndims, 0)))

  return Independent

def BatchSlice(batch_slice: Tuple):
  """A higher-order operation on Distributions that slices on their parameter
  arrays. This effectively marginalizes out batch distributions."""
  def slicey_slice(dist: Distribution) -> Distribution:
    # For annoying reasons, it's not possible to have Distribution subtype
    # NamedTuple even though all distributions are also NamedTuples. But we need
    # the input here to be Iterable. There's not a clear way to specify that the
    # input must be a Distribution *and* a NamedTuple AFAICT.
    params_broadcasted = jp.broadcast_arrays(*cast(Iterable, dist))
    return dist.__class__(*[arr[batch_slice] for arr in params_broadcasted])

  return slicey_slice

def DiagMVN(loc: jp.ndarray, scale: jp.ndarray) -> Distribution:
  return Independent(1)(Normal(loc, scale))

class MVN(Distribution, NamedTuple):
  loc: jp.ndarray
  scale_tril: jp.ndarray

  @property
  def event_shape(self):
    return (self.loc.shape[-1], )

  @property
  def batch_shape(self):
    # loc.shape       should be [..., d]
    # self.scale_tril shoule be [..., d, d]
    # broadcasted...  should be [..., d]
    return lax.broadcast_shapes(self.loc.shape, self.scale_tril[:-1])[:-1]

  def sample(self, rng, sample_shape=()) -> jp.ndarray:
    z = random.normal(rng, shape=sample_shape + self.event_shape)
    return self.loc + jp.inner(z, self.scale_tril)

  def log_prob(self, x) -> jp.ndarray:
    (d, ) = self.event_shape

    # Put a new dimension at the end to force solve_triangular to treat delta as
    # a matrix with shape [..., d, 1]. Otherwise batch dimensions could alter
    # the behavior and produce weird effects.
    delta = x - self.loc
    delta = delta[..., jp.newaxis]

    renorm = linalg.solve_triangular(self.scale_tril, delta, lower=True)**2
    # renorm will have shape [..., d, 1] so we need to sum over the last two
    # dimensions.
    return (d * NEG_HALF_LOG_TWO_PI - self._logdet_scale_tril -
            0.5 * jp.sum(renorm, axis=(-2, -1)))

  def entropy(self) -> jp.ndarray:
    (d, ) = self.event_shape
    return d / 2 - d * NEG_HALF_LOG_TWO_PI + self._logdet_scale_tril

  @property
  def _logdet_scale_tril(self):
    # The determinant of a triangular matrix is the product of its diagonal.
    return jp.sum(jp.log(jp.diagonal(self.scale_tril, axis1=-2, axis2=-1)),
                  axis=-1)
