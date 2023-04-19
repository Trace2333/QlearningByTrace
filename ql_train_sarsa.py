import gymnasium as gym
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate, plt_out

use_render_human = False
use_slippery = True
policy = "sarsa"
episodes = 3000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0
epsilon_decay = 1e-3

logging.basicConfig(level=logging.INFO)


if use_render_human:
    if use_slippery:
        environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode='human')
    else:
        environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')
else:
    if use_slippery:
        environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    else:
        environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)

environment.reset()
environment.render()

logging.basicConfig(level=logging.INFO)

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

outcomes = []

print('Q-table before training:')
print(qtable)

# Training

iter_obj = tqdm(range(episodes))

# Training
for s in iter_obj:
    state, _ = environment.reset()
    terminated = False

    iter_obj.set_postfix(step=s)

    outcomes.append("Failure")

    if policy == "sarsa":
        action = np.argmax(qtable[state])

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not terminated:
        # Generate a random number between 0 and 1
        rnd = np.random.random()
        # Implement this action and move the agent in the desired direction
        if policy == "Q":

            if rnd < epsilon:
                action = environment.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            observation, reward, terminated, truncated, info = environment.step(action)

            qtable[state, action] = qtable[state, action] + \
                        alpha * (reward + gamma * np.max(qtable[observation]) - qtable[state, action])

            state = observation
        elif policy == "sarsa":

            observation, reward, terminated, truncated, info = environment.step(action)

            if rnd < epsilon:
                action_new = environment.action_space.sample()
            else:
                action_new = np.argmax(qtable[observation])

            qtable[state, action] = qtable[state, action] + \
                                    alpha * (reward + gamma * qtable[observation][action_new] - qtable[state, action])

            state = observation
            action = action_new

        if reward:
            outcomes[-1] = "Success"

    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)


print('===========================================')
print('Q-table after training:')
print(qtable)

plt_out(outcomes)

evaluate(environment, qtable)