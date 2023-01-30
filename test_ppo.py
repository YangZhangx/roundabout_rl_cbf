import gym
from stable_baselines3 import PPO
import os
import highway_env
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

models_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('roundabout-v0') 
env.reset()

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name='PPO')
    model.save(f"{models_dir}/{TIMESTEPS*i}")

'''tensorboard_log=logdir'''
'''tb_log_name='PPO'''
'''iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")'''