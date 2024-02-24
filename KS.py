class KnapsackEnv:
    def __init__(self, items, max_weight):
        """
        Initializes the knapsack environment.
        :param items: List of tuples where each tuple is (value, weight) for each item.
        :param max_weight: Maximum weight capacity of the knapsack.
        """
        self.items = items
        self.max_weight = max_weight
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.total_value = 0
        self.current_weight = 0
        self.available_items = list(range(len(self.items)))  # Indexes of items available for picking
        self.state = (self.current_weight, tuple(self.available_items))  # Initial state
        return self.state

    def step(self, action):
        if action not in self.available_items:
            reward = -10  # Penalty for choosing an invalid action
            done = False
            info = {'msg': "Invalid action."}
        else:
            item_value, item_weight = self.items[action]
            if self.current_weight + item_weight <= self.max_weight:
                self.current_weight += item_weight
                self.total_value += item_value
                self.available_items.remove(action)
                reward = item_value  # Reward is the value of the item added
                info = {'msg': "Item added."}
            else:
                reward = -10  # Penalty for exceeding weight limit
                info = {'msg': "Exceeded weight limit."}
        
        done = not self.available_items or self.current_weight == self.max_weight
        self.state = (self.current_weight, tuple(self.available_items))
        return self.state, reward, done, info

    def render(self):
        """
        Optional: Prints the current state of the environment for debugging/visualization.
        """
        print(f"Total Value: {self.total_value}, Current Weight: {self.current_weight}, Available Items: {self.available_items}")


import numpy as np

class QLearningAgent:
    def __init__(self, action_size, state_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(available_actions)  # Explore
        else:
            q_values = self.q_table[state, available_actions]
            return available_actions[np.argmax(q_values)]  # Exploit

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])  # Best action for next state
        td_target = reward + self.discount_rate * self.q_table[next_state, best_next_action] * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Update exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def learn(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                available_actions = env.available_items
                action = self.choose_action(state, available_actions)
                next_state, reward, done, info = env.step(action)
                self.update_q_table(state, action, reward, next_state, done)
                state = next_state
            if episode % 100 == 0:
                print(f"Episode {episode}: Exploration Rate {self.exploration_rate}")



class KnapsackEnvSimplified(KnapsackEnv):
    def reset(self):
        super().reset()
        # Simplify state to just the current weight for this example
        self.state = self.current_weight
        return self.state

    def step(self, action):
        # Simplified to update and return just the current weight as state
        state, reward, done, info = super().step(action)
        self.state = self.current_weight  # Simplified state representation
        return self.state, reward, done, info



# Assume KnapsackEnv and QLearningAgent classes are defined as previously described

# Initialize the Knapsack environment
items = [(60, 10), (100, 20), (120, 30)]  # Example items: (value, weight)
max_weight = 50
env = KnapsackEnvSimplified(items, max_weight)

# Initialize the Q-Learning Agent
# Note: Adjust action_size and state_size based on your environment's requirements
action_size = len(items)  # Max number of actions equals the number of items
state_size = (max_weight + 1) * (2 ** len(items))  # Simplistic state size calculation; adjust as needed
agent = QLearningAgent(action_size, state_size)

episodes = 100  # Reduced for debugging
max_steps_per_episode = 1000  # Add a limit to steps per episode to avoid infinite loops

for episode in range(episodes):
    state = env.reset()
    done = False
    steps = 0  # Track the number of steps

    while not done and steps < max_steps_per_episode:
        action = agent.choose_action(state, env.available_items)
        next_state, reward, done, info = env.step(action)
        agent.update_q_table(state, action, reward, next_state, done)
        state = next_state
        steps += 1  # Increment step count

        # Debugging print statements
        print(f"Episode {episode}, Step {steps}, State {state}, Action {action}, Reward {reward}, Done {done}")

    if done:
        print(f"Episode {episode} finished after {steps} steps.")
    else:
        print(f"Episode {episode} hit the step limit.")

