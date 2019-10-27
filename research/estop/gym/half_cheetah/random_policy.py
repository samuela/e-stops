import time

import gym

env = gym.make("HalfCheetah-v3")
for _ in range(10):
  tic = time.time()
  env.reset()
  for _ in range(1000):
    # env.render()
    env.step(env.action_space.sample())
  print(f"episode {time.time() - tic}")
env.close()
