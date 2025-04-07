import gymnasium as gym
def print_env_info(env):
    print("🔍 Environment Info")
    print("=" * 40)

    # 观测空间
    obs_space = env.observation_space
    print(f"Observation space type: {type(obs_space)}")
    print(f"Observation shape: {obs_space.shape}")
    if isinstance(obs_space, gym.spaces.Box):
        print(f"Observation dtype: {obs_space.dtype}")
        print(f"Observation range: [{obs_space.low.min()}, {obs_space.high.max()}]")

    print("-" * 40)

    # 动作空间
    act_space = env.action_space
    print(f"Action space type: {type(act_space)}")
    if isinstance(act_space, gym.spaces.Discrete):
        print(f"Action space: Discrete({act_space.n})")
    elif isinstance(act_space, gym.spaces.Box):
        print(f"Action shape: {act_space.shape}")
        print(f"Action dtype: {act_space.dtype}")
        print(f"Action range: [{act_space.low.min()}, {act_space.high.max()}]")

    print("-" * 40)

    # 其他环境元数据
    print(f"Max episode steps (if defined): {env.spec.max_episode_steps if env.spec else 'N/A'}")
    print(f"Reward threshold (if defined): {env.spec.reward_threshold if env.spec else 'N/A'}")
    print("=" * 40)
