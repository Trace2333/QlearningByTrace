import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')
environment.reset()
environment.render()

qtable = np.zeros((16, 4))

# Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n      # = 4
qtable = np.zeros((nb_states, nb_actions))

# Let's see how it looks
print('Q-table =')
print(qtable)

random.choice(["LEFT", "DOWN", "RIGHT", "UP"])

environment.action_space.sample()

environment.step(2)
environment.render()

# 1. Randomly choose an action using action_space.sample()
action = environment.action_space.sample()

# 2. Implement this action and move the agent in the desired direction
new_state, reward, done, info, _ = environment.step(action)

# Display the results (reward and map)
environment.render()
print(f'Reward = {reward}')

