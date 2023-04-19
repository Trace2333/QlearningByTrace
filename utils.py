import numpy as np
from matplotlib import pyplot as plt


def evaluate(environment, qtable):
    episodes = 100
    nb_success = 0

    # Evaluation
    for _ in range(100):
        state, _ = environment.reset()
        terminated = False

        # Until the agent gets stuck or reaches the goal, keep training it
        while not terminated:
            # Choose the action with the highest value in the current state
            action = np.argmax(qtable[state])

            # Implement this action and move the agent in the desired direction
            observation, reward, terminated, truncated, info = environment.step(action)

            # Update our current state
            state = observation

            # When we get a reward, it means we solved the game
            nb_success += reward

    # Let's check our success rate!
    print(f"Success rate = {nb_success / episodes * 100}%")


def plt_out(outcomes):
    # Plot outcomes
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
    plt.show()