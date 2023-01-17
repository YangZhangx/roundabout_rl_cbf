import random
import gym
import keras
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.001
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_actor(self):
        # Neural Net for Actor Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.actor_learning_rate))
        return model
    def _build_critic(self):
        # Neural Net for Critic Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.critic_learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, size=(self.action_size,))
        act_values = self.actor.predict(state)
        return act_values
    
    def train(self, batch_size):
        inputs = keras.layers.concatenate([state_input, action_input])
        net = keras.layers.Dense(...) (inputs)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * self.critic.predict(next_state)[0][0]
            action_values = self.actor.predict(state)
            critic_value = self.critic.predict(inputs)

            critic_value = np.array([[target]])
            self.actor.fit(state, action_values, epochs=1, verbose=0)
            self.critic.fit([state, action_values], critic_value, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
if __name__ == '__main__':
    env = gym.make("roundabout-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DDPGAgent(state_size, action_size)
    batch_size = 32
    done = False
    for e in range(1000):
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, info, _  = env.step(action)
            reward = reward if not done else -10
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/1000, score:, e: {:.2}"
                      .format(e, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.train(batch_size)


