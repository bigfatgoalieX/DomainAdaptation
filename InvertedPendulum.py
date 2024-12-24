import gymnasium as gym
from stable_baselines3 import PPO

# 创建 MuJoCo 环境，例如 InvertedPendulum
env = gym.make('InvertedPendulum-v5',render_mode ='human')

# 初始化强化学习模型（PPO）
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=100)

# 测试模型
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

# 关闭环境
env.close()
