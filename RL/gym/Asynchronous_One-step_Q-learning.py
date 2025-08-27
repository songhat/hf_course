import gymnasium as gym
import torch
import numpy as np
import threading
import queue
import time
from torch import nn, optim
from network import MLP_QNet

class GlobalQLearningAgent:
    """
    全局Q-learning智能体，维护共享的Q网络参数。
    - 负责全局Q网络的参数存储与同步。
    - 提供线程安全的参数更新接口。
    - 定期同步目标网络。
    """
    def __init__(self, state_size=8, action_size=4, device='cuda', target_update_freq=1000):
        """
        初始化全局智能体
        输入: 状态空间维度, 动作空间维度, 设备, 目标网络同步频率
        输出: 无
        """
        self.GAMMA = 0.9
        self.LR = 0.0001
        self.ACTION_SIZE = action_size
        self._target_update_freq = target_update_freq
        self._global_step = 0
        self._best_reward = float('-inf')
        self._best_params = None

        self.qnet = MLP_QNet(input_size=state_size, out_size=action_size)
        self.target_qnet = MLP_QNet(input_size=state_size, out_size=action_size)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.LR)
        self.lossFn = nn.MSELoss()
        
        self.device = device
        if torch.cuda.is_available() and device == 'cuda':
            self.qnet = self.qnet.cuda()
            self.target_qnet = self.target_qnet.cuda()
        
        self.lock = threading.Lock()  # 全局参数更新锁

    def update_global_network_params(self, new_params):
        """
        使用新参数更新全局网络
        输入: new_params (OrderedDict) - 新的网络参数
        输出: 无
        """
        with self.lock:
            self.qnet.load_state_dict(new_params)
            self._global_step += 1
            
            # 定期同步目标网络
            if self._global_step % self._target_update_freq == 0:
                self.target_qnet.load_state_dict(self.qnet.state_dict())

    def get_network_params(self):
        """
        获取当前网络参数（线程安全）
        输入: 无
        输出: 网络参数的深拷贝
        """
        with self.lock:
            return self.qnet.state_dict().copy()

    def update_best_model(self, reward, save_path="best_model.pth"):
        """
        更新最佳模型参数
        输入: reward (float) - 当前评估奖励, save_path (str) - 保存路径
        输出: 是否更新了最佳模型 (bool)
        """
        with self.lock:
            if reward > self._best_reward:
                self._best_reward = reward
                self._best_params = self.qnet.state_dict().copy()
                # 保存最佳模型到文件
                torch.save(self._best_params, save_path)
                return True
            return False

    def getAction(self, state, epsilon=0.0):
        """
        根据当前状态选择动作，用于评估
        输入: state (np.array) - 当前环境状态, epsilon (float) - 探索率
        输出: action (int) - 选择的动作
        """
        try:
            state = torch.FloatTensor(state).to(self.device)
            if np.random.rand() < epsilon:
                return np.random.choice(self.ACTION_SIZE)
            else:
                with torch.no_grad():
                    qs = self.qnet(state)
                    return qs.argmax().item()
        except Exception as e:
            print(f"全局智能体动作选择错误: {e}")
            return np.random.choice(self.ACTION_SIZE)

class LocalQLearningAgent:
    """
    本地Q-learning智能体，每个线程拥有独立的agent。
    - 维护本地Q网络副本和累积梯度。
    - 每次交互后计算梯度并累积。
    - 按指定步长使用累积梯度更新本地网络，然后推送参数到全局网络。
    """
    def __init__(self, global_agent, state_size=8, action_size=4, update_freq=5, sync_freq=100):
        """
        初始化本地智能体
        输入: 全局智能体引用, 状态空间维度, 动作空间维度, 梯度累积步长, 同步频率
        输出: 无
        """
        self.global_agent = global_agent
        self.EPSILON = np.random.rand()  # 随机贪婪探索率
        self.ACTION_SIZE = action_size
        self._update_freq = update_freq  # 本地网络更新频率
        self._update_step = 0            # 当前累积的梯度数量
        self._sync_freq = sync_freq      # 同步训练频率
        self._sync_step = 0              # 训练步长
        self.GAMMA = 0.98
        self.lossFn = nn.MSELoss()
        
        # 创建本地网络副本和优化器
        self.local_qnet = MLP_QNet(input_size=state_size, out_size=action_size)
        self.local_optimizer = optim.RMSprop(self.local_qnet.parameters(), lr=global_agent.LR)
        
        if torch.cuda.is_available() and global_agent.device == 'cuda':
            self.local_qnet = self.local_qnet.cuda()


    def getAction(self, state):
        """
        根据当前状态选择动作，使用epsilon-greedy策略
        输入: state (np.array) - 当前环境状态
        输出: action (int) - 选择的动作
        """
        try:
            state = torch.FloatTensor(state).to(self.global_agent.device)
            if np.random.rand() < self.EPSILON:
                return np.random.choice(self.ACTION_SIZE)
            else:
                with torch.no_grad():
                    qs = self.local_qnet(state)
                    return qs.argmax().item()
        except Exception as e:
            print(f"动作选择错误: {e}")
            return np.random.choice(self.ACTION_SIZE)

    def accumulate_gradient(self, state, action, reward, next_state, done):
        """
        计算单步TD误差的梯度并累积到本地缓冲区
        输入: state, action, reward, next_state, done - 单步经验
        输出: loss值
        """
        try:
            state = torch.FloatTensor(state).to(self.global_agent.device)
            next_state = torch.FloatTensor(next_state).to(self.global_agent.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.global_agent.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.global_agent.device)

            # 计算当前Q值
            qs = self.local_qnet(state)
            q = qs[action]
            
            # 计算TD目标
            with torch.no_grad():
                nextQs = self.global_agent.qnet(next_state)
                nextQ = nextQs.max()
                target = (1 - done) * self.GAMMA * nextQ + reward

            # 计算损失
            loss = self.lossFn(q, target)
            # 反向传播计算梯度（不清零梯度，保持累积）
            loss.backward()

            self._update_step += 1
            self._sync_step += 1

            return loss.item()
            
        except Exception as e:
            print(f"梯度累积错误: {e}")
            return 0.0
    
    def update_gradient(self):
        """
        使用累积梯度更新本地网络，然后推送参数到全局网络
        输入: 无
        输出: 是否成功更新和推送
        """
        # 使用平均梯度更新本地网络
        self.local_optimizer.step()
        # 清零梯度，为下一轮累积做准备
        self.local_optimizer.zero_grad()
        self._update_step = 0

    def push_gradients_to_global(self):
        self.global_agent.qnet.load_state_dict(self.local_qnet.state_dict())

def train_worker(global_agent, env_fn, loss_queue, worker_id, episodes=200):
    """
    工作线程函数，运行独立的本地智能体进行异步学习
    输入: 全局智能体, 环境创建函数, 损失队列, 工作线程ID, 训练回合数, 梯度累积步长
    输出: 无（结果写入队列）
    """
    try:
        local_agent = LocalQLearningAgent(global_agent)
        env = env_fn()
        
        for episode in range(episodes):
            state, info = env.reset()
            terminated = False
            truncated = False
            episode_loss = 0
            step_count = 0
            
            while not (terminated or truncated):
                # 选择动作
                action = local_agent.getAction(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # 累积梯度
                loss = local_agent.accumulate_gradient(state, action, reward, next_state, terminated or truncated)
                episode_loss += loss
                step_count += 1

                # 检查是否需要推送梯度
                if local_agent._sync_step % local_agent._sync_freq == 0:
                    local_agent.push_gradients_to_global()
                if local_agent._update_step % local_agent._update_freq == 0 \
                    or (terminated or truncated):
                    local_agent.update_gradient()

                state = next_state
            average_loss = episode_loss / step_count if step_count > 0 else 0
            loss_queue.put((worker_id, episode, average_loss))
            
    except Exception as e:
        print(f"工作线程 {worker_id} 错误: {e}")

def make_env():
    return gym.make("LunarLander-v3", render_mode=None)

def evaluation_worker(global_agent, env_fn, reward_queue, eval_episodes=10, eval_interval=50):
    """
    评估线程函数，定期测试全局智能体性能并保存最佳模型
    输入: 全局智能体, 环境创建函数, 奖励队列, 评估回合数, 评估间隔
    输出: 无（结果写入队列）
    """
    try:
        eval_count = 0
        while True:
            time.sleep(eval_interval)  # 等待指定间隔
            
            total_reward = 0.0
            successful_episodes = 0
            
            # 运行多个评估回合
            for episode in range(eval_episodes):
                env = env_fn()
                state, info = env.reset()
                episode_reward = 0.0
                terminated = False
                truncated = False
                step_count = 0
                max_steps = 1000  # 防止无限循环
                
                while not (terminated or truncated) and step_count < max_steps:
                    # 使用贪婪策略（epsilon=0）进行评估
                    action = global_agent.getAction(state, epsilon=0.0)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    state = next_state
                    step_count += 1
                
                total_reward += episode_reward
                if episode_reward > 200:  # LunarLander成功着陆的奖励阈值
                    successful_episodes += 1
                
                env.close()
            
            # 计算平均奖励
            average_reward = total_reward / eval_episodes
            success_rate = successful_episodes / eval_episodes
            
            # 更新最佳模型
            is_best = global_agent.update_best_model(average_reward)
            
            eval_count += 1
            print(f"评估 #{eval_count}: 平均奖励={average_reward:.2f}, 成功率={success_rate:.2%}, "
                  f"{'新纪录!' if is_best else ''}")
            
            # 将评估结果放入队列
            reward_queue.put({
                'eval_count': eval_count,
                'average_reward': average_reward,
                'success_rate': success_rate,
                'is_best': is_best,
                'best_reward': global_agent._best_reward
            })
            
    except Exception as e:
        print(f"评估线程错误: {e}")

def monitor_training(reward_queue, total_duration=3600):
    """
    监控训练过程的线程函数
    输入: 奖励队列, 总训练时长（秒）
    输出: 无
    """
    start_time = time.time()
    eval_history = []
    
    try:
        while time.time() - start_time < total_duration:
            try:
                # 非阻塞获取评估结果
                result = reward_queue.get(timeout=1.0)
                eval_history.append(result)
                
                # 每10次评估输出一次训练摘要
                if result['eval_count'] % 10 == 0:
                    recent_rewards = [r['average_reward'] for r in eval_history[-10:]]
                    recent_avg = np.mean(recent_rewards)
                    print(f"\n=== 训练摘要 (评估#{result['eval_count']}) ===")
                    print(f"最近10次平均奖励: {recent_avg:.2f}")
                    print(f"历史最佳奖励: {result['best_reward']:.2f}")
                    print(f"训练时长: {(time.time() - start_time)/60:.1f}分钟")
                    print("=" * 40)
                
            except queue.Empty:
                continue
    
    except Exception as e:
        print(f"监控线程错误: {e}")
    
    # 保存评估历史
    if eval_history:
        import json
        with open('training_history.json', 'w') as f:
            json.dump(eval_history, f, indent=2)
        print(f"训练历史已保存到 training_history.json")

if __name__ == "__main__":
    NUM_WORKERS = 4
    EPISODES_PER_WORKER = 250
    UPDATE_STEP = 10
    EVAL_INTERVAL = 30  # 每30秒评估一次
    EVAL_EPISODES = 5   # 每次评估5个回合
    TRAINING_DURATION = 1800  # 训练30分钟
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建全局智能体和队列
    global_agent = GlobalQLearningAgent(state_size=8, action_size=4, device=device)
    loss_queue = queue.Queue()
    reward_queue = queue.Queue()
    
    # 启动训练线程
    worker_threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(
            target=train_worker, 
            args=(global_agent, make_env, loss_queue, i, EPISODES_PER_WORKER),
            daemon=True
        )
        t.start()
        worker_threads.append(t)
    
    # 启动评估线程
    eval_thread = threading.Thread(
        target=evaluation_worker,
        args=(global_agent, make_env, reward_queue, EVAL_EPISODES, EVAL_INTERVAL),
        daemon=True
    )
    eval_thread.start()
    
    # 启动监控线程
    monitor_thread = threading.Thread(
        target=monitor_training,
        args=(reward_queue, TRAINING_DURATION),
        daemon=True
    )
    monitor_thread.start()
    
    print(f"开始异步训练：{NUM_WORKERS}个工作线程，每{UPDATE_STEP}步更新")
    print(f"评估设置：每{EVAL_INTERVAL}秒评估{EVAL_EPISODES}个回合")
    print(f"预计训练时长：{TRAINING_DURATION/60:.1f}分钟")
    
    # 等待训练完成或超时
    start_time = time.time()
    try:
        while time.time() - start_time < TRAINING_DURATION:
            # 检查工作线程是否仍在运行
            active_workers = sum(1 for t in worker_threads if t.is_alive())
            if active_workers == 0:
                print("所有工作线程已完成")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("用户中断训练")
    
    print("训练结束，正在收集最终结果...")
    
    # 收集loss历史
    loss_history = []
    while not loss_queue.empty():
        try:
            worker_id, episode, loss = loss_queue.get_nowait()
            loss_history.append(loss)
        except queue.Empty:
            break
    
    # 最终统计
    if loss_history:
        print(f"\n=== 最终训练统计 ===")
        print(f"总训练回合数: {len(loss_history)}")
        print(f"平均loss: {np.mean(loss_history):.4f}")
        print(f"最佳模型奖励: {global_agent._best_reward:.2f}")
        print(f"模型已保存至: best_model.pth")
    else:
        print("未收集到训练数据")