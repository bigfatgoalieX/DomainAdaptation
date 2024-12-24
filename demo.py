import gymnasium as gym

# 创建 MuJoCo 环境，例如 HalfCheetah
env = gym.make('InvertedPendulum-v5',render_mode='human')

# 重置环境并获得初始状态
state, info = env.reset()

done = False
while not done:
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作并获取反馈
    next_state, reward, done, truncated, info = env.step(action)
    
    # 渲染环境
    env.render()

    # 输出当前状态和奖励
    print(f"State: {next_state}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

# 关闭环境
env.close()
