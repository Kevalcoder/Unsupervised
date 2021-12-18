# Keval and Smit assignment
import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

class SimplePolicyNetwork:
    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 2.0
        self.gamma = .87
        self.batch_size = 72
        self.epsilon_min = .01
        self.lr = 0.005
        self.epsilon_decay = .666
        self.memory = deque(maxlen=10000)
        self.model = self.model()

    def model(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.state_space, activation=relu))
        model.add(Dense(150, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        new_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        states = np.squeeze(states)
        new_states = np.squeeze(new_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(new_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_simplepolicy(episode):

    loss = []
    agent = SimplePolicyNetwork(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 2000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        is_solved = np.mean(loss[-100:])
        if is_solved > 150:
            print("Task Completed ")
            break
        print("The Average of last 100 episodes: {0:.2f} \n".format(is_solved))
    return loss


if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 15
    #Provide the parameters of the worst and best network policy and plot their corresponding Mean reward on the y-axis to Episode on the x-axis
    loss = train_simplepolicy(episodes)
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()