#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    for i in range(10):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            if dones:
                break
            env.render()

        env.close()