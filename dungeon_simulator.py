from enums import *
import random

class DungeonSimulator:
    def __init__(self, length=5, slip=0.1, small=2, large=10):
        self.length = length # Length of the dungeon
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for BACKWARD action
        self.large = large  # payout at end of chain for FORWARD action
        self.state = 0  # Start at beginning of the dungeon

    def take_action(self, action):
        if random.random() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action == BACKWARD:  # BACKWARD: go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif action == FORWARD:  # FORWARD: go up along the dungeon
            if self.state < self.length - 1:
                self.state += 1
                reward = 0
            else:
                reward = self.large
        return self.state, reward

    def reset(self):
        self.state = 0  # Reset state to zero, the beginning of dungeon
        return self.state