import random
import json
import argparse
import time
from drunkard import Drunkard
from accountant import Accountant
from gambler import Gambler
from deep_gambler import DeepGambler
from dungeon_simulator import DungeonSimulator

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='GAMBLER', help='Which agent to use')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='How quickly the algorithm tries to learn')
    parser.add_argument('--discount', type=float, default=0.98, help='Discount for estimated future action')
    parser.add_argument('--iterations', type=int, default=2000, help='Iteration count')
    ARGS, unparsed = parser.parse_known_args()

    # select agent
    if ARGS.agent == 'GAMBLER':
        agent = Gambler(learning_rate=ARGS.learning_rate, discount=ARGS.discount, iterations=ARGS.iterations)
    elif ARGS.agent == 'ACCOUNTANT':
        agent = Accountant()
    elif ARGS.agent == 'DEEPGAMBLER':
        agent = DeepGambler(learning_rate=ARGS.learning_rate, discount=ARGS.discount, iterations=ARGS.iterations)
    else:
        agent = Drunkard()

    # setup simulation
    dungeon = DungeonSimulator()
    dungeon.reset()
    total_reward = 0 # Score keeping
    last_total = 0

    # main loop
    for step in range(ARGS.iterations):
        old_state = dungeon.state # Store current state
        action = agent.get_next_action(old_state) # Query agent for the next action
        new_state, reward = dungeon.take_action(action) # Take action, get new state and reward
        agent.update(old_state, new_state, action, reward) # Let the agent update internals

        total_reward += reward # Keep score
        if step % 250 == 0: # Print out metadata every 250th iteration
            performance = (total_reward - last_total) / 250.0
            print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
            last_total = total_reward

        time.sleep(0.0001) # Avoid spamming stdout too fast!

    # print("Final Q-table", agent.q_table)

if __name__ == "__main__":
    main()
