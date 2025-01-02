import sys
import gymnasium
sys.modules["gym"] = gymnasium
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from graspGymEnv import graspGymEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EveryNTimesteps, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp

date = datetime. now(). strftime("%Y%m%d-%I:%M%p")

env = graspGymEnv()
# evalenv = Monitor(env)
# trainenv = DummyVecEnv([lambda: env])

NAME = f"{date}_SAC"
callback = EvalCallback(env, best_model_save_path=f"./models/{NAME}/", log_path=f"./logs/{NAME}/", eval_freq=80_000, deterministic=True)#, callback_after_eval=stop_train_callback, n_eval_episodes=10)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard/{NAME}/", device="cpu")

model.learn(total_timesteps=50_000_000, callback=callback, log_interval=20)

model.save(f"./tensorboard/{NAME}/model")