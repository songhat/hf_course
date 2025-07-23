import os
from pyvirtualdisplay import Display

# 启动虚拟显示器
display = Display(visible=0, size=(1400, 900))
display.start()

# 测试OpenGL渲染
try:
    import gymnasium as gym
    
    # 创建一个简单的环境来测试渲染
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    env.reset()
    
    # 尝试渲染一帧
    frame = env.render()
    print(f"渲染成功！帧大小: {frame.shape}")
    
    env.close()
except Exception as e:
    print(f"渲染测试失败: {e}")