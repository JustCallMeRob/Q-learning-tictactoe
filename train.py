import numpy as np
import random
from environment import Environment
import pickle


# Start the Q learning algorithm with the given parameters
def initiate_q_learning(environment,
                        nr_of_epochs,
                        nr_of_episodes,
                        max_steps_per_episode,
                        learning_rate,
                        discount_rate,
                        exploration_rate,
                        max_exploration_rate,
                        min_exploration_rate,
                        exploration_decay_rate):
    # Initialize the Q tables for both agents with all the values set to zero
    q_table_agent_x = np.zeros((environment.state_space, environment.action_space))
    q_table_agent_o = np.zeros((environment.state_space, environment.action_space))

    # initialize epoch reward dictionary
    epoch_rewards = {}

    for epoch in range(nr_of_epochs):

        # reset the exploration rate for current epoch
        current_exploration_rate = exploration_rate
        # save all the rewards for every episode of the current epoch for each respective agent
        all_rewards_agent_x = []
        all_rewards_agent_o = []

        for episode in range(nr_of_episodes):

            # reset the environment back to default
            state = environment.reset()
            # flag to show if an agent has won and the other lost, or its a draw
            end_state = False
            # reset reward for current episode
            current_reward_agent_x = 0
            current_reward_agent_o = 0

            for step in range(max_steps_per_episode):

                # Take step with agent X
                # exploration-exploitation trade-off
                exploration_threshold = random.uniform(0, 1)
                # if threshold is larger than the rate (initially 1), then exploit the environment
                if exploration_threshold > current_exploration_rate:
                    # we exploit the environment by choosing the biggest q value action for the current state
                    action = np.argmax(q_table_agent_x[state, :])
                # else explore the environment
                else:
                    # we explore the environment by making a random move and observing the amount of reward
                    action, end_state = environment.sample()

                # only true if no valid moves are possible anymore, a tie was reached
                if end_state:
                    environment.render()
                    current_reward_agent_x += -10
                    current_reward_agent_o += 10
                    break

                # take the next action
                new_state, reward, end_state = environment.step(action, 1)

                # update Q table for Q(s, a) (using the formula for q^new, see link)
                q_table_agent_x[state, action] = q_table_agent_x[state, action] * \
                                                 (1 - learning_rate) + learning_rate * \
                                                 (reward + discount_rate * np.max(q_table_agent_x[new_state, :]))

                state = new_state
                current_reward_agent_x += reward

                # if end state was reached on agent X's turn, it means agent O lost
                if end_state:
                    environment.render()
                    current_reward_agent_o += -10
                    break

                # Take step with agent O
                # exploration-exploitation trade-off
                exploration_threshold = random.uniform(0, 1)
                # if threshold is larger than the rate (initially 1), then exploit the environment
                if exploration_threshold > current_exploration_rate:
                    # we exploit the environment by choosing the biggest Q value action for the current state
                    action = np.argmax(q_table_agent_o[state, :])
                # else explore the environment
                else:
                    # we explore the environment by making a random move and observing the amount of reward
                    action, end_state = environment.sample()

                # take the next action
                new_state, reward, end_state = environment.step(action, 2)

                # update Q table for Q(s, a) (using the formula for q^new, see link)
                q_table_agent_o[state, action] = q_table_agent_o[state, action] * \
                                                 (1 - learning_rate) + learning_rate * \
                                                 (reward + discount_rate * np.max(q_table_agent_o[new_state, :]))

                state = new_state
                current_reward_agent_o += reward

                # if end state was reached on agent O's turn, it means agent X lost
                if end_state:
                    environment.render()
                    current_reward_agent_x += -10
                    break

            # decay the exploration rate, using exponential decay
            current_exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
                                       np.exp(-exploration_decay_rate * episode)

            # append current rewards to all rewards for respective agents
            all_rewards_agent_x.append(current_reward_agent_x)
            all_rewards_agent_o.append(current_reward_agent_o)

        epoch_rewards[epoch] = [all_rewards_agent_x, all_rewards_agent_o]

    # Print updated Q table
    print("\n\nQ-table")
    print(q_table_agent_x)
    print(q_table_agent_o)
    # Save Q tables
    pickle.dump(q_table_agent_x, open('./q_tables/q_table_agent_x.p', 'wb'))
    pickle.dump(q_table_agent_o, open('./q_tables/q_table_agent_o.p', 'wb'))
    print("Number of unique states explored: ", environment.state_counter, " out of ", Environment.state_space)
    calculate_average_reward(epoch_rewards, nr_of_episodes)
    return q_table_agent_x, q_table_agent_o


# Calculate average rewards per ten thousand episodes
def calculate_average_reward(epoch_rewards, nr_of_episodes):
    for epoch, rewards in epoch_rewards.items():
        print("EPOCH: ", epoch)
        print("Average reward per 10000 episodes for agent X")
        rewards_per_thousand_episodes = np.split(np.array(rewards[0]), nr_of_episodes / 10000)
        count = 10000
        for r in rewards_per_thousand_episodes:
            print(count, ": ", str(sum(r / 10000)))
            count += 10000
        print("Average reward per 10000 episodes for agent O")
        rewards_per_thousand_episodes = np.split(np.array(rewards[1]), nr_of_episodes / 10000)
        count = 10000
        for r in rewards_per_thousand_episodes:
            print(count, ": ", str(sum(r / 10000)))
            count += 10000