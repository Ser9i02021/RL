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
        self.max_steps = max_steps if max_steps is not None else self.T * self.J
        self.state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        self.state = np.zeros((self.J, self.T), dtype=int)    # state = [0, ..., 0] for each job and time step
        self.steps_taken = 0
        return self.state.flatten()
    
    def step(self, action):
        # "action" is to insert a job at a time step, or remove it if it's already there
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        
        # Toggle the job state at the given time step (activate if inactive, deactivate if active)
        self.state[job, time_step] = 1 - self.state[job, time_step]
        
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps or np.all(np.sum(self.state, axis=1) == 1)
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

class DoubleDQNAgent:
    def __init__(self, n_inputs, n_actions):
        self.policy_net = DQN(n_inputs)
        self.target_net = DQN(n_inputs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.n_actions = n_actions
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



def train(agent, env, episodes=2000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.append(Experience(torch.tensor([state], dtype=torch.float),
                                           torch.tensor([[action]], dtype=torch.long),
                                           torch.tensor([next_state], dtype=torch.float),
                                           torch.tensor([reward], dtype=torch.float)))
            state = next_state
            total_reward += reward
            agent.optimize_model()
            if done:
                break
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_net()
        #print(f"Episode {episode+1}, Total Reward: {total_reward}")

job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

env = ONTSEnv(job_priorities, energy_consumption, max_energy)
agent = DoubleDQNAgent(n_inputs=env.J * env.T, n_actions=env.J * env.T)
train(agent, env, episodes=2000)

def evaluate(agent, env, episodes=10):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards += episode_reward
        print(f"Episode {episode+1}: Reward: {episode_reward}")
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")

evaluate(agent, env, episodes=10)
