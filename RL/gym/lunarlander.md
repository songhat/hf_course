# REINFORCE算法
训练周期：5000个回合

策略网络:

```python
class PolicyNet(nn.Module):

    def __init__(self, action_size: int):
        super().__init__()

        self.l1 = nn.LazyLinear(64)
        self.l2 = nn.LazyLinear(128)
        self.l3 = nn.LazyLinear(128)
        self.l4 = nn.LazyLinear(action_size)

    def forward(self, x: torch.Tensor) 
        x = F.dropout(F.relu(self.l1(x)))
        x = F.dropout(F.relu(self.l2(x)))
        x = F.dropout(F.relu(self.l3(x)))
        x = F.softmax(self.l4(x), dim=-1)
        return x
```
reinforce_lunar_lander.pth
![alt text](image.png)

利用layer norm层来优化数据的分布，使得训练稳定。LN层的位置是参考pytorch里面的transformer模型里LN的默认位置。
```python
class PolicyNet(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
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
```

reinforce_lunar_lander2.pth
![alt text](image-1.png)
