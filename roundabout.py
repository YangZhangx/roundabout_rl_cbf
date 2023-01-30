import gym
import os
import numpy as np
import highway_env
import keras
from matplotlib import pyplot as plt
#%matplotlib inline
#from stable_baselines import HER, SAC, PP02, DQN
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.vec_env import DummyVecEnv


#env_name = "roundabout-v0"
#env = gym.make(env_name)

env = DummyVecEnv([lambda: gym.make('roundabout-v0')])

#ddpg = DDPG('MlpPolicy', 'roundabout-v0', verbose=1)
#ddpg.learn(total_timesteps=int(1e5))

#model = SAC('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps=int(1e5))



model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))

obs = env.reset()
obs = np.reshape(obs, (1, -1))
