import gym
import highway_env
import numpy as np

from stable_baselines3 import HER, SAC, DDPG, TD3
#from stable_baselines3 import NormalActionNoise
highway_env.register_highway_envs()
env = gym.make("parking-v0")
env.reset()
env.render()
n_sampled_goal = 4

model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future', verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layer=[256, 256, 256]))

model.learn(int(2e5))
model.save('her_sac_highway')

obs = env.reset()

# 100次的reward作为评价指标
episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward

    if done or info.get('is_success', False):
        print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        episode_reward = 0.0
        obs = env.reset()

