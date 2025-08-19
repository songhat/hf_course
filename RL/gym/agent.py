import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self, input_size=8, out_size=4):
        super(QNet, self).__init__()
        self.l1 = nn.Linear(input_size, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, out_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

# 简单的QLearning算法
class QLearningAgent:
    def __init__(self, state_size=8, action_size=4, device='cuda'):
        self.gamma = 0.9
        self.lr = 0.0001
        self.epsilon = 0.1
        self.action_size = action_size

        self.qnet = QNet(input_size=state_size, out_size=action_size)
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
        self.lossFn = nn.MSELoss()

        self.device = device

    def getAction(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if np.random.rand() < self.epsilon: # e-greedy
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                qs = self.qnet(state)
                return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        qs = self.qnet(state) # todo 理解为什么get_action 和这里要分开获取qs
        q = qs[action]

        with torch.no_grad():
            nextQs = self.qnet(next_state)
            nextQ = nextQs.max()
            target = (1 - done) * self.gamma * nextQ + reward # TD算法

        loss = self.lossFn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self._buffer)

    def get_batch(self):
        batch = random.sample(self._buffer, self._batch_size)
        state = np.stack([x[0] for x in batch])
        action = np.array([x[1] for x in batch])
        reward = np.array([x[2] for x in batch])
        next_state = np.stack([x[3] for x in batch])
        done = np.array([x[4] for x in batch]).astype(np.float32)
        return state, action, reward, next_state, done

class DQNAgent:
    def __init__(self, state_size=8, action_size=4, device='cuda'):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = action_size

        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(state_size, action_size).to(self._device)
        self.qnet_target = QNet(state_size, action_size).to(self._device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self._device)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                qs = self.qnet(state)
                return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self._replay_buffer.add(state, action, reward, next_state, done)
        if len(self._replay_buffer) < self.batch_size:
            return 0.0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._replay_buffer.get_batch()
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        action_batch = torch.LongTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        done_batch = torch.FloatTensor(done_batch).to(self._device)

        qs = self.qnet(state_batch)
        q = qs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.qnet_target(next_state_batch)
            next_q = next_qs.max(dim=1)[0]
            target = reward_batch + (1 - done_batch) * self.gamma * next_q

        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())