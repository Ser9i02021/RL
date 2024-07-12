# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import torch.nn.functional as F


# %%
class KnapsackEnv:
    def __init__(self, weights, values, max_weight, max_steps=None):
        self.weights = weights
        self.values = values
        self.max_weight = max_weight
        self.max_steps = max_steps if max_steps is not None else len(weights)
        self.n = len(weights)   # number of objects
        self.state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        self.state = np.zeros(self.n, dtype=int)    # state = [0, ..., 0]
        self.steps_taken = 0
        return self.state
    
    def step(self, action):
        # "action" is to insert an object in the KS, if it is not in it
        # or to take an object from the KS otherwise
        self.steps_taken += 1
        if self.state[action] == 0:
            self.state[action] = 1
        else:
            self.state[action] = 0
        reward, weight_exceeded = self.calculate_reward()
        done = weight_exceeded or self.steps_taken >= self.max_steps
        return self.state, reward, done
    
    def calculate_reward(self):
        weight = np.sum(self.state * self.weights)  # dot product of "state" and "weights" vectors
        value = np.sum(self.state * self.values)    # dot product of "state" and "values" vectors
        if weight > self.max_weight:
            return -100, True  # Penalize for exceeding max weight
        return value, False


# %%
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, n_inputs):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_inputs)  # Output size = number of actions
        )
    
    def forward(self, x):
        return self.fc(x)

def train_dqn(env, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.n
    policy_net = DQN(n_actions)
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        t = 0
        while True:  # sem numero maximo, numero maximo definido no env
            epsilon = max(epsilon * eps_decay, eps_end)
            action = select_action(state, policy_net, epsilon, n_actions)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            
            memory.append(Experience(torch.tensor([state], dtype=torch.float),
                                     torch.tensor([[action]], dtype=torch.long),
                                     torch.tensor([next_state], dtype=torch.float),
                                     torch.tensor([reward], dtype=torch.float)))
            
            state = next_state

            optimize_model(policy_net, optimizer, memory, gamma, batch_size)
            t=t+1
            if done:
                episode_durations.append(t + 1)
                break
        #print(f"Episode {episode+1}, Total Reward: {total_reward}")

    return policy_net  # return best policy


def select_action(state, policy_net, epsilon, n_actions):
    # Return Value: The method returns the selected action as an integer,
    # which indicates the object to insert into or remove from the knapsack in the environment.
    # (index of "state = [0, 1, ..., 1, 0]")

    # Epsilon-Greedy Strategy: This method uses an epsilon-greedy strategy to balance
    # exploration and exploitation. With this strategy, the agent chooses actions in two ways:
    # Exploration: With probability epsilon, the agent explores the environment by choosing
    # an action at random. This allows the agent to try new actions that it has not evaluated
    # extensively yet.
    # Exploitation: With probability 1 - epsilon, the agent exploits its current knowledge
    # of the environment by choosing the action that it believes has the highest expected reward
    # according to the current policy network (policy_net). This is done by predicting the
    # Q-values for all possible actions from the current state and selecting the action with
    # the highest Q-value.
    sample = random.random()
    if sample > epsilon:
        # For Exploitation: The state is converted into a tensor and passed through
        # the policy network to get the predicted Q-values for all actions from that state.
        # The action with the maximum Q-value is selected.
        with torch.no_grad():
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        # For Exploration: A random action is selected from the range of possible actions
        # (0 to n_actions-1) using random.randrange(n_actions).
        return random.randrange(n_actions)

def optimize_model(policy_net, optimizer, memory, gamma, batch_size): #*****
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    # batch concatenation
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    
    # computed the Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    # Change for Huber loss, got lazy to search the function in docs *****
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %%
weights = np.array([1, 2, 3, 2, 2])  # Weights of items
values = np.array([6, 10, 12, 8, 7])  # Values of items
max_weight = 10  # Maximum weight that the knapsack can carry

env = KnapsackEnv(weights, values, max_weight) # 1st - create the environmnet


# %%
policy_net = train_dqn(env, episodes=1000)  # 2nd - Train for 1000 episodes

# %%
def evaluate_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        #policy_net = train_dqn(env, episodes=1000)  # 2nd - Train for 1000 episodes
        state = env.reset()
        episode_reward = 0
        while True:
            action = select_action(state, policy_net, epsilon=0, n_actions=env.n)  # Use epsilon=0 for greedy action selection (using only the net policy built through its trainining)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done: # max weight exceeded or max num_steps taken
                break
        total_rewards += episode_reward
        print(f"Episode {episode+1}: Reward: {episode_reward}")
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")


evaluate_model(env, policy_net) # 3rd - Evaluate the trained model



