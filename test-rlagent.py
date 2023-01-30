# Environment
import gym
import highway_env

# Agent
#pip install git+https://github.com/eleurent/rl-agents#egg=rl-agents
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm.notebook import trange
#!pip install gym pyvirtualdisplay
#!apt-get install -y xvfb python-opengl ffmpeg
#!git clone https://github.com/eleurent/highway-env.git
sys.path.insert(0, './highway-env/scripts/')
from utils import record_videos, show_videos



# Make environment
env = gym.make("roundabout-v0")
env = record_videos(env)
(obs, info), done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    
env.close()
show_videos()