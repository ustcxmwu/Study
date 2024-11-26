#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   rl_1.py
@Time    :   2024-11-22 09:50
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""

import gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    # 创建 Gym 环境
    env = gym.make("CartPole-v1")

    # 创建模型：使用 PPO 算法
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练模型
    model.learn(total_timesteps=10000)

    # 保存模型
    model.save("ppo_cartpole")

    # 加载模型
    model = PPO.load("ppo_cartpole")

    # 测试训练好的模型
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
