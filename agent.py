import numpy as np

from action_value_table import ActionValueTable

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3  # agents actions
GAMMA = 0.95
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        if done:
            for i in self.actions:
                self.values.set_value(next_state, i, 0)
        
        current_value = self.values.get_value(state, action)
        max_value = []
        for j in self.actions:
            max_value.append(self.values.get_value(next_state, j))
        max_value = max(max_value)
        new_value = current_value + (self.step_size * (reward + (self.gamma * max_value) - current_value))
        self.values.set_value(state, action, new_value)
        
        return


    def get_action(self, state):
        value_list = []
        for a in self.actions:
            value_list.append(self.values.get_value(state, a))
        max_value = np.max(value_list)
        
        max_count = 0
        for j in self.actions:
            if self.values.get_value(state, j) == max_value:
                max_count += 1
        
        probabilities = []
        for i in self.actions:
            if self.values.get_value(state, i) == max_value:
                # Implement max value probability
                probabilities.append(((1 - self.epsilon)/max_count) + (self.epsilon/len(self.actions)))
            else:
                # Implement not max value probability
                probabilities.append(self.epsilon/len(self.actions))
        
        return np.random.choice(self.actions, p = probabilities)


    def get_greedy_action(self, state):
        value_list = []
        for a in self.actions:
            value_list.append(self.values.get_value(state, a))
        max_value = np.max(value_list)

        max_list = []
        for j in self.actions:
            if self.values.get_value(state, j) == max_value:
                max_list.append(j)
        
        return np.random.choice(max_list)