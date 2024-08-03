import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque
from torch_geometric.data import Data, Batch
import pickle
from torch_geometric.nn import GATConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # global mean pooling
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x





class ONTSEnv:
    def __init__(self, job_priorities, energy_consumption, max_energy, max_steps=None):
        self.job_priorities = job_priorities
        self.energy_consumption = energy_consumption
        self.max_energy = max_energy
        self.J, self.T = energy_consumption.shape
        self.max_steps = max_steps if max_steps is not None else self.T
        self.state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        self.state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        return self.state.flatten()
    
    def step(self, action):
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        self.state[job, time_step] = 1 - self.state[job, time_step]
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        for t in range(self.T):
            totalEnergyAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyAtTimeStep_t += self.state[j][t] * self.energy_consumption[j][t]
            
            if totalEnergyAtTimeStep_t > self.max_energy:
                return -100, True  # Penalize for exceeding max energy
            
        rewardSum = 0
        for j in range(self.J):
            for t in range(self.T):
                # Reward directly proportional to the priority and inversely proportional to the energy consumption on that job at that time
                rewardSum += (self.job_priorities[j] * self.state[j][t]) * (self.max_energy + 1 - self.energy_consumption[j][t])

        
        if np.all(np.sum(self.state, axis=1) == 1):  # If each job is scheduled exactly once
            return rewardSum, False
        else:        
            return -1, False  # Scale down the reward if not all jobs are scheduled


    def get_graph(self):
        edge_index = self.create_edges()
        x = torch.tensor(self.state.flatten(), dtype=torch.float).view(-1, 1)
        data = Data(x=x, edge_index=edge_index)
        return data

    def create_edges(self):
        edges = []
        for job in range(self.J):
            for t in range(self.T - 1):
                edges.append((job * self.T + t, job * self.T + t + 1))
                edges.append((job * self.T + t + 1, job * self.T + t))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index




def train_gnn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.J * env.T
    policy_net = GNN(in_channels=1, out_channels=n_actions) if pn is None else pn
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000) if mem is None else mem
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)
            action = select_action_gnn(env, policy_net, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            memory.append(Experience(env.get_graph(), torch.tensor([[action]], dtype=torch.long), env.get_graph(), torch.tensor([reward], dtype=torch.float)))
            optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size)
            if done:
                episode_durations.append(total_reward)
                break
    return policy_net, memory

def select_action_gnn(env, policy_net, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            state_graph = env.get_graph()
            q_values = policy_net(state_graph)
            return q_values.max(1)[1].item()
    else:
        return random.randrange(env.J * env.T)
    

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))



def optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))
    
    state_batch = Batch.from_data_list([exp for exp in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = Batch.from_data_list([exp for exp in batch.next_state])

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




def evaluate_gnn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = select_action_gnn(env, policy_net, epsilon=0)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
        print(f"Episode {episode+1}: Reward: {reward}")
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")





job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

env = ONTSEnv(job_priorities, energy_consumption, max_energy)
env.reset()

# Initial training
policy_net, memory = train_gnn(env, episodes=1000)

# Store GNN policy and memory into a binary file
with open('policy.txt', 'wb') as file:
    pickle.dump(policy_net, file)
with open('mem.txt', 'wb') as file:
    pickle.dump(memory, file)
'''
# Retrieve GNN policy and memory and store them into variables
with open('policy.txt', 'rb') as file:
    policy_net = pickle.load(file)
with open('mem.txt', 'rb') as file:
    memory = pickle.load(file)

# Continue the GNN training with the current net policy and memory
#for _ in range(10):
policy_net, memory = train_gnn(env, policy_net, memory, episodes=1000)

# Store GNN policy and memory into a binary file
with open('policy.txt', 'wb') as file:
    pickle.dump(policy_net, file)
with open('mem.txt', 'wb') as file:
    pickle.dump(memory, file)
'''

evaluate_gnn_model(env, policy_net, episodes=10)

