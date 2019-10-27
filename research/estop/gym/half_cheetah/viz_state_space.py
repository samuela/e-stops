import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import tqdm
import numpy as np

from research.estop.gym.half_cheetah import env_name

env = gym.make(env_name)

delta = 2
env.reset()
mjstate = env.sim.get_state()
qpos = mjstate.qpos

for i in tqdm.trange(len(qpos)):
  video = VideoRecorder(env, path=f"qpos_{i}_sweep.mp4")

  a = qpos - delta * np.eye(len(qpos))[i]
  b = qpos + delta * np.eye(len(qpos))[i]
  for t in np.linspace(0, 1):
    env.sim.set_state(mjstate._replace(qpos=(1 - t) * a + t * b))
    env.sim.step()
    video.capture_frame()

  video.close()
