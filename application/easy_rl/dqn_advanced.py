#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
@Project :   Study
@File    :   dqn_advanced.py
@Time    :   2025-01-20 14:40
@Author  :   Xiaomin Wu <wuxiaomin@pandadagames.com>
@Version :   1.0
@License :   (C)Copyright 2025, Xiaomin Wu
@Desc    :   chapter 8
"""

import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from application.easy_rl import rl_utils


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN(object):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device,
        dqn_type="VanillaDQN",
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.dqn_type = dqn_type
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.q_net(state)
            return torch.argmax(q_value).item()

    def max_q_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_value = self.q_net(state)
        return torch.max(q_value).item()

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float32).to(self.device)
        actions = (
            torch.tensor(batch["actions"], dtype=torch.int64)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(batch["rewards"], dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32).to(
            self.device
        )
        dones = (
            torch.tensor(batch["dones"], dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[0].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action / (action_dim - 1)) * (
        action_upbound - action_lowbound
    )


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=0)
                episode_over = False
                while not episode_over:
                    action = agent.take_action(state)
                    max_q_value = (
                        agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    )  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                    episode_over = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, episode_over)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list, max_q_value_list


def main():
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 将连续动作分成11个离散动作

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DQN(
        state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device
    )
    return_list, max_q_value_list = train_DQN(
        agent, env, num_episodes, replay_buffer, minimal_size, batch_size
    )

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("DQN on {}".format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c="orange", ls="--")
    plt.axhline(10, c="red", ls="--")
    plt.xlabel("Frames")
    plt.ylabel("Q value")
    plt.title("DQN on {}".format(env_name))
    plt.show()
    #

if __name__ == "__main__":
    main()
