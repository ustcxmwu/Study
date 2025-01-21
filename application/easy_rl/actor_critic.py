#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
@Project :   Study
@File    :   actor_critic.py
@Time    :   2025-01-20 16:00
@Author  :   Xiaomin Wu <wuxiaomin@pandadagames.com>
@Version :   1.0
@License :   (C)Copyright 2025, Xiaomin Wu
@Desc    :
"""

import gym
import torch
import torch.nn.functional as F
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCritic(object):
    def __init__(
        self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.int64)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )

        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_targets - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_error.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_targets.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


def main():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(
        state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device
    )

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)


if __name__ == "__main__":
    main()
