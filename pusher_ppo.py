import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import print_env_info

# 超参数
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 3e-4
UPDATE_EPOCHS = 10
MAX_EPISODES = 500
BATCH_SIZE = 2048

# 策略 + 值网络（共享主体）
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable std
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.base(obs)
        return self.mean_head(x), self.value_head(x)

    def get_action(self, obs):
        mean, _ = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(self, obs, action):
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze()

# GAE or Return计算（这里简化为累计回报）
def compute_returns(rewards, dones, values, gamma=GAMMA):
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

# PPO更新
def ppo_update(model, optimizer, obs, actions, log_probs_old, returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

    for _ in range(UPDATE_EPOCHS):
        new_log_probs, entropy, new_values = model.evaluate(obs, actions)
        ratio = torch.exp(new_log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(new_values, returns)
        entropy_bonus = entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 主程序
def main():
    env = gym.make("Pusher-v5", render_mode="human")
    print_env_info(env);
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for episode in range(MAX_EPISODES):
        obs_buf, act_buf, logprob_buf, reward_buf, done_buf, val_buf = [], [], [], [], [], []

        obs, _ = env.reset()
        
        tip_pos = obs[14:17]
        object_pos = obs[17:20]
        goal_pos = obs[20:23]

        print(f"🖐 Tip position   : {tip_pos}")
        print(f"📦 Object position: {object_pos}")
        print(f"🎯 Goal position  : {goal_pos}")
        
        done = False
        total_reward = 0
        steps = 0

        while len(obs_buf) < BATCH_SIZE:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            value = model.forward(obs_tensor)[1].detach().item()
            action, log_prob, _ = model.get_action(obs_tensor)
            action_np = action.detach().numpy()
            action_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated

            obs_buf.append(obs_tensor)
            act_buf.append(torch.tensor(action_np, dtype=torch.float32))
            logprob_buf.append(log_prob.detach())
            reward_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)

            total_reward += reward
            steps += 1
            obs = next_obs

            if done:
                obs, _ = env.reset()

        returns = compute_returns(reward_buf, done_buf, val_buf)
        ppo_update(
            model,
            optimizer,
            torch.stack(obs_buf),
            torch.stack(act_buf),
            torch.stack(logprob_buf),
            returns,
            torch.tensor(val_buf)
        )

        print(f"Episode {episode}, Total reward: {total_reward:.2f}, Steps: {steps}")

    env.close()

if __name__ == "__main__":
    main()
