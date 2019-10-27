def DistributionLayer(dist):
  def init_fn(_rng, input_shape):
    # Distributions are just flat (named) tuples, so the output will also be a
    # tuple with the same shape.
    return input_shape, None

  def apply_fn(_params, inputs, **_):
    return dist(*inputs)

  return init_fn, apply_fn
