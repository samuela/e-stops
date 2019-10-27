import numpy as np

# def softmax(x, axis=None):
#   exp = np.exp(x)
#   return exp / np.sum(exp, axis=axis, keepdims=True)

def softmax(x, axis=None):
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)
