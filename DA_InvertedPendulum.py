import gymnasium as gym
import random
from stable_baselines3 import PPO

# 自定义一个环境，继承自 InvertedPendulum-v5，并在每个重置时应用随机化
class RandomizedInvertedPendulumEnv(gym.Env):
    def __init__(self, env_name='InvertedPendulum-v5'):
        super().__init__()
        self.env = gym.make(env_name)
        self.randomize()

    def randomize(self):
        # 随机化摩擦系数，质量等物理参数
        self.env.unwrapped.sim.model.geom_friction[0] = [random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 0.5)]
        self.env.unwrapped.sim.model.geom_friction[1] = [random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 0.5)]
        self.env.unwrapped.sim.model.option.gravity = [random.uniform(-10, -9.5), 0, random.uniform(-9.9, -9.81)]  # 随机化重力

    def reset(self, **kwargs):
        self.randomize()  # 每次重置时随机化环境
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# 创建源环境A（包含域随机化）
env_A = RandomizedInvertedPendulumEnv(env_name='InvertedPendulum-v5')

# 创建目标环境B（与环境A有所不同）
env_B = gym.make('InvertedPendulum-v5', render_mode='human')  # 可以是原始环境或其他不同设置

# 初始化强化学习模型（PPO），并在环境A中训练
model = PPO("MlpPolicy", env_A, verbose=1)

# 在环境A中训练模型
model.learn(total_timesteps=10000)

# 测试阶段：在目标环境B中测试
obs = env_B.reset()
done = False
while not done:
    action, _states = model.predict(obs)  # 预测动作
    obs, reward, done, truncated, info = env_B.step(action)  # 执行动作并获得新的状态
    env_B.render()  # 渲染目标环境B

# 关闭目标环境
env_B.close()
