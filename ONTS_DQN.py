# %% [markdown]
# Environment Definition

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import torch.nn.functional as F

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
        self.state = np.zeros((self.J, self.T), dtype=int)    # state = [0, ..., 0] for each job and time step
        self.steps_taken = 0
        return self.state.flatten()
    
    def step(self, action):
        # "action" is to insert a job at a time step, or remove it if it's already there
        job, time_step = divmod(action, self.T) # If the agent selects action 7: divmod(7, 5) will yield (1, 2), meaning job 1 at time step 2
        self.steps_taken += 1
        
        self.state[job, time_step] = 1 - self.state[job, time_step]  # toggle the job state at the given time step
        
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        #print(done)
        #print(reward)
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        total_energy = np.sum(self.state * self.energy_consumption)
        total_priority = np.sum(self.state * self.job_priorities[:, None])
        #print(self.state)
        # Check if the total energy consumption exceeds the maximum allowed energy
        if total_energy > self.max_energy:
            return -100, True  # Penalize for exceeding max energy  
        
        if total_energy ==  self.max_energy:
            return total_priority, True

        return total_priority, False

# %% [markdown]
# DQN Model

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
    
    


# %% [markdown]
# Training the DQN

# %%
def train_dqn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.J * env.T
    if (pn == None):
        policy_net = DQN(n_actions)
    else:
        policy_net = pn
    optimizer = optim.Adam(policy_net.parameters())
    if (mem == None):
        memory = deque(maxlen=10000)
    else:
        memory = mem
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        t = 0
        while True:
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
            t = t + 1
            if done:
                episode_durations.append(t + 1)
                break

    return policy_net, memory

def select_action(state, policy_net, epsilon, n_actions):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(n_actions)

def optimize_model(policy_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %% [markdown]

# %% [markdown]
# Evaluating the Model

# %%
def evaluate_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = select_action(state, policy_net, epsilon=0, n_actions=env.J * env.T)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
        print(f"Episode {episode+1}: Reward: {reward}")
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")


# Running the Training

# %%
job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

env = ONTSEnv(job_priorities, energy_consumption, max_energy)
env.reset()
policy_net, memory = train_dqn(env, episodes=4000)
policy_net, memory = train_dqn(env, policy_net, memory, episodes=4000)



evaluate_model(env, policy_net, episodes=10)



