import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F

# Pointer Network Implementation
class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PointerNet, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.pointer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_outputs, _ = self.decoder(x, (hidden, cell))
        pointer_logits = self.pointer(decoder_outputs)
        return pointer_logits

# Experience namedtuple
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Define the environment
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
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        
        self.state[job, time_step] = 1 - self.state[job, time_step]  # toggle the job state at the given time step
        
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        total_energy = np.sum(self.state * self.energy_consumption)
        total_priority = np.sum(self.state * self.job_priorities[:, None])
        if total_energy > self.max_energy:
            return -100, True  # Penalize for exceeding max energy  
        return total_priority, False

# Training function
def train_pointer_net(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.J * env.T
    if (pn == None):
        policy_net = PointerNet(env.J * env.T, 128)
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
            
            # Assuming state, next_state are lists of NumPy arrays and action, reward are single values
            # Convert state and next_state to a single tensor using torch.stack
            state_tensor = torch.tensor(np.stack(state), dtype=torch.float)
            next_state_tensor = torch.tensor(np.stack(next_state), dtype=torch.float)

            # Append to memory using the tensors
            memory.append(Experience(state_tensor,
                                    torch.tensor([[action]], dtype=torch.long),  # Assuming action is a scalar
                                    next_state_tensor,
                                    torch.tensor([reward], dtype=torch.float)))  # Assuming reward is a scalar

            
            state = next_state

            optimize_model(policy_net, optimizer, memory, gamma, batch_size)
            t = t + 1
            if done:
                episode_durations.append(t + 1)
                break

    return policy_net, memory

# Select action
def select_action(state, policy_net, epsilon, n_actions):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(np.stack(state), dtype=torch.float).unsqueeze(0)  # Add batch dimension
            pointer_logits = policy_net(state_tensor)
            action_probs = F.softmax(pointer_logits.view(-1), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            return action
    else:
        return random.randrange(n_actions)

# Optimize model
def optimize_model(policy_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    state_batch = torch.cat(batch.state).view(batch_size, 1, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state).view(batch_size, 1, -1)
    
    state_action_values = policy_net(state_batch).view(batch_size, -1).gather(1, action_batch)
    next_state_values = policy_net(next_state_batch).view(batch_size, -1).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluating the Model
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
job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

env = ONTSEnv(job_priorities, energy_consumption, max_energy)
env.reset()
policy_net, memory = train_pointer_net(env, episodes=100)
policy_net, memory = train_pointer_net(env, policy_net, memory, episodes=100)

# Evaluating the model
evaluate_model(env, policy_net, episodes=10)
