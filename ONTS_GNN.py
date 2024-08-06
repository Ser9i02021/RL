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
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e, max_steps=None):
        self.u__job_priorities = u__job_priorities
        self.q__energy_consumption_per_job = q__energy_consumption_per_job
        self.y_min_per_job = y_min_per_job
        self.y_max_per_job = y_max_per_job
        self.t_min_per_job = t_min_per_job
        self.t_max_per_job = t_max_per_job
        self.p_min_per_job = p_min_per_job
        self.p_max_per_job = p_max_per_job
        self.w_min_per_job = w_min_per_job
        self.w_max_per_job = w_max_per_job
        self.r__energy_available_at_time_t = r__energy_available_at_time_t
        self.gamma = gamma 
        self.Vb = Vb
        self.Q = Q
        self.p = p
        self.e = e
        self.SoC_t = self.p

        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T
        self.x__state = None
        self.phi__state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        self.x__state = np.zeros((self.J, self.T), dtype=int)
        self.phi__state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        self.SoC_t = self.p
        return self.x__state.flatten()
    
    def step(self, action):
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]
        self.build_phi_matrix() # Auxiliary matrix to check constraints
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.x__state, reward, done
    
    def build_phi_matrix(self):
        for j in range(self.J):
            for t in range(self.T):
                if t == 0:
                    if self.x__state[j, t] > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1
                else:
                    if (self.x__state[j, t] - self.x__state[j, t-1]) > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1
                    if self.phi__state[j, t] > (2 - self.x__state[j, t] - self.x__state[j, t-1]):
                        self.phi__state[j, t] = 0
                
                if self.phi__state[j, t] > self.x__state[j, t]:
                    self.phi__state[j, t] = 0
                

    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                #print("Penalize for exceeding max energy available at that time step!")
                return -1, False  # Penalize for exceeding max energy available at that time step
            
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)
            #print("SoC: %f" % (self.SoC_t))

            if self.SoC_t > 1:
                #print("Penalize for exceeding max state of charge at that time step")
                return -1, False  # Penalize for exceeding max state of charge at that time step
            
        #print("All energy constraints satisfied!")
        return 0, False

    def check_job_constraints(self):
        acc_reward = 0
        for j in range(self.J):
            
            # w_min, w_max ((w_min, w_max] is the interval a job can be active)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    #print("Penalize for activating job %d at a disallowed time step (bellow min: %d)" % (j + 1, self.w_min_per_job[j] + 1))
                    acc_reward -= 1  # Penalize for activating a job at a disallowed time step (bellow min)
                
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    #print("Penalize for activating job %d at a disallowed time step (above max: %d)" % (j + 1, self.w_max_per_job[j]))
                    acc_reward -= 1  # Penalize for activating a job at a disallowed time step (above max)
                
            
            # y_min, y_max (Min and Max times a job can be active)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                #print("Penalize for job %d not having been executed at least 'y_min_per_job': %d times" % (j + 1, self.y_min_per_job[j]))
                acc_reward -= 1   # Penalize for a job not having been executed at least "y_min_per_job" times
            if sum_l > self.y_max_per_job[j]:
                #print("Penalize for job %d having been executed more than 'y_max_per_job': %d times" % (j + 1, self.y_max_per_job[j]))
                acc_reward -= 1   # Penalize for a job having been executed more than "y_max_per_job" times
            

            # t_min, t_max (continuous job execution) 
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt] 
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    #print("Penalize for job %d not running continuosly for its minimum period to do that: %d" % (j + 1, self.t_min_per_job[j]))
                    acc_reward -= 1   # Penalize for a job not running continuosly for its minimum period to do that 
            
            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    #print("Penalize for job %d running continuosly more than its maximum period to do that: %d" % (j + 1, self.t_max_per_job[j]))
                    acc_reward -= 1   # Penalize for a job running continuosly more than its maximum period to do that
                

            for t in range(self.T - self.t_min_per_job[j] + 1, self.T):
                sum_l = 0
                for l in range(t, self.T):
                    sum_l += self.x__state[j, l]
                if sum_l < (self.T - t + 1) * self.phi__state[j, t]:
                    #print("Penalize for job %d..." % (j + 1))
                    acc_reward -= 1   # Penalize for a job...
                
            
            # p_min, p_max (periodic job execution)            
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    #print("Penalize for job %d not having been executed periodically for at least every 'p_min_per_job': %d time steps" % (j + 1, self.p_min_per_job[j]))
                    acc_reward -= 1   # Penalize for a job not having been executed periodically for at least every "p_min_per_job" time steps
                
            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    #print("Penalize for job %d having been executed periodically for more than every 'p_max_per_job': %d time steps" % (j + 1, self.p_max_per_job[j]))
                    acc_reward -= 1   # Penalize for a job having been executed periodically for more than every "p_max_per_job" time steps
    
        
        #if acc_reward == 0:
            #print("All job constraints satisfied!")
        
        return acc_reward
            
    
    def calculate_reward(self):
        rewardSum = 0
        reward, done = self.check_energy_constraints()
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # Reward directly proportional to the priority and inversely proportional to the energy consumption of that job
                    # Plus, it is downgraded if at least one restriction is not met
                    if rewardSum == 0:
                        rewardSum += (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])

            return rewardSum, False
        
        return rewardSum, done


    def get_graph(self):
        edge_index = self.create_edges()
        x = torch.tensor(self.x__state.flatten(), dtype=torch.float).view(-1, 1)
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
            '''if reward > 0:
                print("state:")
                print(next_state)
                #print(env.phi__state)
                print("episode: %d; reward: %f" % (episode, reward))
                print()'''
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




# Creating an instance for the ONTS

# Job prioirities
u__job_priorities = np.array([3, 2, 1])

# Energy consumption for every time a job is activated
q__energy_consumption_per_job = np.array([1, 2, 1])

# Min and max times a job can execute
y_min_per_job = [1, 1, 1] 
y_max_per_job = [3, 4, 5]

# Min and max periods for continuos execution per job
t_min_per_job = [1, 1, 1]
t_max_per_job = [3, 4, 3]

# Min and max periodic execution time steps per job
p_min_per_job = [1, 1, 1]
p_max_per_job = [4, 5, 5]

# Min and max time step a job is allowed to run
w_min_per_job = [0, 0, 0]
w_max_per_job = [4, 5, 4]

# Max energy solar panel can spend at a given time step
r__energy_available_at_time_t = np.array([3, 3, 3, 3, 3])

# Scaling factor for energy consumption
gamma = 0.5

# Battery voltage
Vb = 1

# Battery capacity
Q = 10

# Min state of charge
p = 0.1

# State of charge initialization
#SoC = p

# Discharge efficiency
e = 0.9

env = ONTSEnv(u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e)
env.reset()
'''
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


evaluate_gnn_model(env, policy_net, episodes=10)
