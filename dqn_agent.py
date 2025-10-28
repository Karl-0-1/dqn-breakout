import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A tuple to store one transition (experience)
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Flattened size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, num_actions, buffer_capacity, batch_size, gamma, eps_start, eps_end, eps_decay, lr, target_update_freq):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        
        self.policy_net = QNetwork(num_actions).to(device)
        self.target_net = QNetwork(num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        self.steps_done = 0

    def _state_to_tensor(self, state):
        """Converts a state (LazyFrames or np.array) to a tensor."""
        state_np = np.array(state)
        state_np_squeezed = np.squeeze(state_np, axis=-1)
        return torch.tensor(state_np_squeezed, device=device, dtype=torch.float32).unsqueeze(0)

    def select_action(self, state, exploration=True):
        sample = random.random()
        
        # Calculate current epsilon
        if exploration:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
        else:
            eps_threshold = 0.0 # Greedy action for evaluation
        
        if sample > eps_threshold:
            # Exploitation
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                action_values = self.policy_net(state_tensor)
                return action_values.max(1)[1].view(1, 1)
        else:
            # Exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=device, dtype=torch.bool
        )
        
        non_final_next_states_list = [
            self._state_to_tensor(s) for s in batch.next_state if s is not None
        ]
        
        if non_final_next_states_list:
            non_final_next_states = torch.cat(non_final_next_states_list)
        else:
            non_final_next_states = torch.empty(0, 4, 84, 84, device=device)

        state_batch = torch.cat([self._state_to_tensor(s) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            if non_final_next_states_list:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_q_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(torch.load(path, map_location=device))
