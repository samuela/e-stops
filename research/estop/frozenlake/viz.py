import matplotlib.pyplot as plt
import numpy as np

from research.estop.frozenlake.frozenlake import Lake

def plot_heatmap(lake: Lake, heat1d):
  im = plt.imshow(lake.reshape(heat1d))

  # Add lake tile labels.
  for i in range(lake.width):
    for j in range(lake.height):
      tile = lake.lake_map[i, j]
      if tile != "F":
        im.axes.text(
            j, i, tile, {
                "horizontalalignment": "center",
                "verticalalignment": "center",
                "color": "white"
            })

  return im

def plot_errorfill(x, ys, color):
  mean = np.mean(ys, axis=0)
  std = np.std(ys, axis=0)
  #   plt.plot(x, mean, color=color, alpha=1.0)
  plt.plot(x, np.median(ys, axis=0), color=color, alpha=1.0)
  plt.fill_between(x,
                   np.maximum(mean - std, ys.min(axis=0)),
                   np.minimum(mean + std, ys.max(axis=0)),
                   color=color,
                   alpha=0.25)
