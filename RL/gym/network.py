import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_QNet(nn.Module):
    # 输出的是Q值，回归任务
    def __init__(self, action_size: int):
        super().__init__()
        # 使用 LazyLinear 以适配未知输入维度（与 Chainer 的自动推断等价）
        self.l1 = nn.LazyLinear(64)
        self.l2 = nn.LazyLinear(128)
        self.l3 = nn.LazyLinear(256)
        self.l4 = nn.LazyLinear(action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.l1(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = F.dropout(F.relu(self.l2(x)))
        x = F.layer_norm(x, x.shape[1:])
        
        x = F.dropout(F.relu(self.l3(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = F.softmax(self.l4(x), dim=-1)
        return x
    
class CNN_QNet(nn.Module):
    def __init__(self, action_size=4):
        super(CNN_QNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # 7x7需根据输入shape调整
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class PolicyNet(nn.Module):
    """
    # 输出的是动作（分类）分布，分类任务
    简单的两层策略网络。
    输入: 张量 (N, D) 或 (D,)；输出: 每个动作的概率 (N, A) 或 (A,)。
    """
    def __init__(self, action_size: int):
        super().__init__()
        # 使用 LazyLinear 以适配未知输入维度（与 Chainer 的自动推断等价）
        self.l1 = nn.LazyLinear(64)
        self.l2 = nn.LazyLinear(128)
        self.l3 = nn.LazyLinear(256)
        self.l4 = nn.LazyLinear(action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.l1(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = F.dropout(F.relu(self.l2(x)))
        x = F.layer_norm(x, x.shape[1:])
        
        x = F.dropout(F.relu(self.l3(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = F.softmax(self.l4(x), dim=-1)
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.LazyLinear(128)
        self.l2 = nn.LazyLinear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
