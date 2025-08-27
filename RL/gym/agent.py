import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torchvision import transforms
from network import MLP_QNet, CNN_QNet, PolicyNet, ValueNet
from torch.distributions import Categorical

class UnsqueezeTransform:
    def __call__(self, x):
        return x.unsqueeze(0) if x.dim() == 3 else x
    
# 简单的QLearning算法
class QLearningAgent:
    def __init__(self, state_size=8, action_size=4, device='cuda'):
        self.gamma = 0.9
        self.lr = 0.0001
        self.epsilon = 0.1
        self.action_size = action_size

        # 使用 MLP_QNet（修复未定义的 QNet 引用），并放到设备上
        self.qnet = MLP_QNet(input_size=state_size, out_size=action_size)
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
        self.lossFn = nn.MSELoss()

        self.device = device
        if torch.cuda.is_available() and device == 'cuda':
            self.qnet = self.qnet.cuda()

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
    def __init__(self, buffer_size, batch_size, use_cnn):
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self.use_cnn = use_cnn

    def add(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self._buffer)

    def get_batch(self, transforms=None): # 这种实现太慢了！
        batch = random.sample(self._buffer, self._batch_size)
        
        action = np.array([x[1] for x in batch])
        reward = np.array([x[2] for x in batch])
        if self.use_cnn:
            state = torch.concat([transforms(x[0]) for x in batch],dim=0)
            next_state = torch.concat([transforms(x[3]) for x in batch],dim=0)
        else:
            state = np.stack([x[0] for x in batch])
            next_state = np.stack([x[3] for x in batch])
        done = np.array([x[4] for x in batch]).astype(np.float32)
        return state, action, reward, next_state, done

class DQNAgent:
    def __init__(self, lr=0.0001, epsilon=0.1, batch_size=32, state_size=8, action_size=4, device='cuda',use_cnn=False,image_shape=(84, 84)):
        self.gamma = 0.98
        self.lr = lr
        self.epsilon = epsilon
        self.buffer_size = 10000
        self.batch_size = batch_size
        self.action_size = action_size

        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_cnn = use_cnn
        self._replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.use_cnn)
        if self.use_cnn:
            QNet = CNN_QNet()
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_shape),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                UnsqueezeTransform()
            ])
        else:
            QNet = MLP_QNet(state_size, action_size)
        self.qnet = QNet.to(self._device)
        self.qnet_target = QNet.to(self._device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        # 如果是CNN，自动做transform
        if self.use_cnn:
            state = self.transforms(state)
            state = state.unsqueeze(0) if state.dim() == 3 else state  # (C,H,W) -> (1,C,H,W)
        else:
            state = torch.FloatTensor(state) if not torch.is_tensor(state) else state
        state = state.to(self._device)
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

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._replay_buffer.get_batch(self.transforms)
        # CNN输入批量transform
        if not self.use_cnn:
            state_batch = torch.FloatTensor(state_batch) if not torch.is_tensor(state_batch) else state_batch
            next_state_batch = torch.FloatTensor(next_state_batch) if not torch.is_tensor(next_state_batch) else next_state_batch

        action_batch = torch.LongTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device)
        done_batch = torch.FloatTensor(done_batch).to(self._device)
        state_batch = state_batch.to(self._device)
        next_state_batch = next_state_batch.to(self._device)

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




class REINFORCE:

    def __init__(self, action_size: int = 2, gamma: float = 0.98, lr: float = 2e-4, device: str | None = None,
                 update_freq=64):
        self.gamma = gamma
        self.lr = lr
        self.action_size = action_size

        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory: list[tuple[float, torch.Tensor]] = []
        self.pi = PolicyNet(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
        self.T = 0
        self.update_freq = update_freq

    def eval(self):
        self.pi.eval()

    def get_action(self, state):
        """给定状态采样一个动作。
        输入: state(ndarray|Tensor) -> (action:int, prob_selected:Tensor)
        """
        state_t = state if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.float32)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        state_t = state_t.to(self.device)
        probs = self.pi(state_t)  # (A,)

        m = Categorical(probs=probs)
        actions = m.sample() 
        probs_selected = probs[torch.arange(probs.size(0)), actions]
        return actions.cpu().numpy(), probs_selected

    def add(self, reward, prob):
        """缓存一条(reward, prob_selected) 轨迹项。"""
        self.memory.append((float(reward), prob))

    def update(self):
        """基于 REINFORCE 目标: L = -sum_t log(pi(a_t|s_t)) * G_t"""
        if not self.memory:
            return 0.0
        

        G = 0.0
        loss = torch.zeros((), device=self.device)
        for reward, prob in reversed(self.memory): # 蒙特卡洛方法
            G = reward + self.gamma * G
            # 数值稳定性，小常数避免 log(0)
            loss = loss + (-torch.log(prob + 1e-8) * G)

        loss.backward()
        self.T += 1
        self.memory = []

        if self.T % self.update_freq == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss.detach().cpu().item())
    

class ActorCritic:
    """
        本质就是REINFORCE的 TD + 价值函数基线方法 的改进
    """
    def __init__(self, lr_pi: float = 0.0002, lr_v: float = 0.0005, action_size: int = 2, gamma: float = 0.98, device: str | None = None):
        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.action_size = action_size

        self._device = device
        self.loss_fn = nn.MSELoss()
        self.pi = PolicyNet(self.action_size).to(self._device)
        self.v = ValueNet().to(self._device)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = state[np.newaxis, :]  # add batch axis
        state = torch.FloatTensor(state) if not torch.is_tensor(state) else state
        state = state.to(self._device)
        probs = self.pi(state)
        probs = probs[0]
        
        m = Categorical(probs=probs)
        action = m.sample().item()
        return action, probs

    def update(self, state, action, action_probs, reward, next_state, done):
        state = state[np.newaxis, :]  # add batch axis
        next_state = next_state[np.newaxis, :]
        
        state = torch.tensor(state, dtype=torch.float32).to(self._device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self._device)
        action_prob = action_probs[action]
        reward = torch.tensor(reward, dtype=torch.float32).to(self._device)
        done = torch.tensor(done, dtype=torch.float32).to(self._device)


        # ========== (1) Update V network ===========
        with torch.no_grad():
            target = reward + self.gamma * self.v(next_state) * (1 - done)
        
        v = self.v(state)
        loss_v = self.loss_fn(v, target)

        # ========== (2) Update pi network ===========
        with torch.no_grad():
            delta = target - v

        loss_pi = -torch.log(action_prob + 1e-8) * delta

        # 分别更新两个网络
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()