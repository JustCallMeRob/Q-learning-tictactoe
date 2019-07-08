from environment import Environment
from train import initiate_q_learning
import sys
import numpy as np


# Simple function to play against one of the agents (not very polished)
def play_against_agent(env, agent):
    state = env.reset()
    print("Introduce the id of the action you want to make, ex: 0 for first cell, 4 for middle cell etc.")
    for step in range(9):
        if agent == 1:
            action = np.argmax(q1[state, :])
            new_state, reward, end_state = env.step(action, 1)
            env.render()
            if end_state:
                if reward == 10:
                    print("YOU LOSE")
                else:
                    print("YOU WIN")
                break
            state = new_state
            action = int(input("Next action: "))
            new_state, reward, end_state = env.step(action, 2)
            env.render()
            if end_state:
                if reward == 10:
                    print("YOU WIN")
                else:
                    print("YOU LOSE")
                break
            state = new_state
        elif agent == 2:
            action = int(input("Next action: "))
            new_state, reward, end_state = env.step(action, 1)
            env.render()
            if end_state:
                if reward == 10:
                    print("YOU WIN")
                else:
                    print("YOU LOSE")
                break
            state = new_state
            action = np.argmax(q2[state, :])
            new_state, reward, end_state = env.step(action, 2)
            env.render()
            if end_state:
                if reward == 10:
                    print("YOU LOSE")
                else:
                    print("YOU WIN")
                break
            state = new_state


if __name__ == '__main__':
    # initialize environment
    env = Environment()
    # number of times we go over all the episodes while resetting the exploration vs exploitation parameters
    nr_of_epochs = 10
    # number of full games the agents will play
    nr_of_episodes = 10000
    # max number of steps that can be taken in one episode
    max_steps_per_episode = 5
    # the rate at which q table values change if we have received new values for states that already had values
    learning_rate = 0.1
    # the rate at which we discount for future state rewards (we prioritise immediate rewards more than future rewards)
    discount_rate = 0.60
    # parameters related to exploitation vs. exploration
    # to begin with, we want our agents to explore more than it exploits
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    # by the end we want our agent to exploit more than it explores, so we decay the exploration rate
    exploration_decay_rate = 0.001

    q1, q2 = initiate_q_learning(env,
                                 nr_of_epochs,
                                 nr_of_episodes,
                                 max_steps_per_episode,
                                 learning_rate,
                                 discount_rate,
                                 exploration_rate,
                                 max_exploration_rate,
                                 min_exploration_rate,
                                 exploration_decay_rate)

    com = input("Which agent would you wish to fight against ? (x,o)")
    if com is "x":
        agent = 1
    elif com is "o":
        agent = 2
    else:
        sys.exit()

    play_against_agent(env, agent)