import numpy as np
import random


class Environment:
    # The game of tic tac toe has a state space of size 5478 with 9 possible actions for any state
    # most states will have less than 9 possible actions, however if the agent decides based on the Q table to make a
    # move on a tile that already has a symbol on it, it is counted as an instant loss
    state_space = 5478
    action_space = 9

    def __init__(self):
        # Initiate the state of the game board to be empty
        # the state of the board will be saved in a 2D numpy array where:
        # 0 = empty cell
        # 1 = X
        # 2 = O
        self.state = np.zeros((3, 3))
        self.state_counter = 0
        self.state_dict = {tuple([0., 0., 0., 0., 0., 0., 0., 0., 0.]): 0}

    # Set the state of the environment to one that we wish
    def set_state(self, state):
        self.state = state
        state_r = self.state.ravel()
        if tuple(state_r) not in self.state_dict:
            self.state_counter += 1
            self.state_dict[tuple(state_r)] = self.state_counter

    # Reset the environments state to default
    def reset(self):
        self.state = np.zeros((3, 3))
        return 0

    # Draw the game board in the current state
    def render(self):
        board = " ___ ___ ___ "
        for row in self.state:
            board += "\n|"
            for cell in row:
                if cell == 1:
                    board += " X "
                elif cell == 2:
                    board += " O "
                else:
                    board += "   "
                board += "|"
        board += "\n ___ ___ ___ \n"
        print(board)

    # Return a random number that determines where on the board the next symbol will be placed
    # only valid empty positions are considered, end state is reached if no such position exists
    def sample(self):
        state_r = np.ravel(self.state)
        valid_positions = 0
        for val in state_r:
            if val == 0:
                valid_positions += 1
        if valid_positions > 0:
            rand = random.randint(1, valid_positions)
        else:
            return 0, True
        valid_positions = 0
        for idx, val in enumerate(state_r):
            if val == 0:
                valid_positions += 1
            if rand == valid_positions:
                return idx, False

    # Take the provided action with the provided agent
    def step(self, action, agent):
        end_state = False
        reward = 0
        state_r = self.state.ravel()

        # check to see if the move is valid
        if state_r[action] != 0:
            end_state = True
            reward = -10
            return self.state_dict[tuple(state_r)], reward, end_state

        # make the move
        state_r[action] = agent
        self.state = np.reshape(state_r, (-1, 3))

        # add the new state to the state dictionary so it can be referenced later in training
        if tuple(state_r) not in self.state_dict:
            self.state_counter += 1
            self.state_dict[tuple(state_r)] = self.state_counter

        # calculate the reward and determine if end state has been reached
        for idx, row in enumerate(self.state):
            # check horizontal
            if np.array_equal(row, [agent, agent, agent]):
                reward = 10
                end_state = True
                return self.state_dict[tuple(state_r)], reward, end_state
            # check vertical
            elif np.array_equal(self.state[:, idx], [agent, agent, agent]):
                reward = 10
                end_state = True
                return self.state_dict[tuple(state_r)], reward, end_state

        # check cross
        if self.state[0, 0] == self.state[1, 1] == self.state[2, 2] == agent:
            reward = 10
            end_state = True
            return self.state_dict[tuple(state_r)], reward, end_state

        # check other cross
        if self.state[0, 2] == self.state[1, 1] == self.state[2, 0] == agent:
            reward = 10
            end_state = True
            return self.state_dict[tuple(state_r)], reward, end_state

        # if the game is not over we reward the agent with -1
        if not end_state:
            reward = -1

        return self.state_dict[tuple(state_r)], reward, end_state