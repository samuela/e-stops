from research.estop.gym.ddpg_training import debug_run, make_default_ddpg_train_config
from research.estop.gym.gym_wrappers import build_env_spec
from research.estop.gym.half_cheetah import env_name, reward_adjustment

env_spec = build_env_spec(env_name, reward_adjustment)
debug_run(env_spec, make_default_ddpg_train_config(env_spec))
