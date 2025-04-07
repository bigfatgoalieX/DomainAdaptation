import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建Ant环境
env = gym.make("Ant-v5", render_mode="human")

# 设置训练参数
policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])  # 策略网络架构

# 初始化PPO模型
model = PPO(
    "MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99,
    policy_kwargs=policy_kwargs, tensorboard_log="./ppo_ant_tensorboard/"
)

# 训练模型
print("Training Ant model...")
model.learn(total_timesteps=100000)  # 训练10万步

# 保存模型
model.save("ppo_ant")

# 加载模型并进行评估
print("Evaluating model...")
saved_model = PPO.load("ppo_ant")
mean_reward, std_reward = evaluate_policy(saved_model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 演示模型效果
def run_trained_model(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()  # 渲染效果
    env.close()

# 运行训练好的模型
run_trained_model(env, saved_model)
