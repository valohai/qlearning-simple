from enums import *
import random

class Drunkard:
    def __init__(self):
        self.q_table = None

    def get_next_action(self, state):
        # Random walk
        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        pass # I don't care! I'm drunk!!