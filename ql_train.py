import gymnasium as gym
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate, plt_out

use_render_human = False
use_slippery = True
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor


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

"""
◀️ LEFT = 0
🔽 DOWN = 1
▶️ RIGHT = 2
🔼 UP = 3
"""

action_table = {0: "LEFT",
                1: "DOWN",
                2: "RIGHT",
                3: "UP"}

outcomes = []

print('Q-table before training:')
print(qtable)

# Training

iter_obj = tqdm(range(episodes))

for s in iter_obj:
    state, _ = environment.reset()
    terminated = False

    iter_obj.set_postfix(step=s)

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not terminated:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

            # logging.info(f"Find action. Do {action_table[action]}")

            iter_obj.set_postfix_str(f"Find action. Do {action_table[action]}")

        # If there's no best action (only zeros), take a random one
        else:
            action = environment.action_space.sample()

            #logging.info(f"no action found, Do {action_table[action]}")

            iter_obj.set_postfix_str(f"no action found, Do {action_table[action]}")

        # Implement this action and move the agent in the desired direction
        observation, reward, terminated, truncated, info = environment.step(action)

        new_state = observation

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)
plt_out(outcomes)
evaluate(environment, qtable)
