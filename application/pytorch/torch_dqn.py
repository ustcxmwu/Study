import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_SIZE = 200

env = gym.make("CartPole-v1", render_mode="human")
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action = self.out(x)
        return action


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_SIZE, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # x = torch.tensor(x[0])
        if np.random.uniform() < EPSILON:
            action = self.eval_net.forward(x)
            # action = torch.max(action, 1)[1].numpy()[0]
            # action = int(torch.max(action).detach().numpy())
            action = int(torch.argmax(action).detach().numpy())
            if action == 2:
                print(action)
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_eps(self, s, a, r, s_):
        eps = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_SIZE
        self.memory[index, :] = eps
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_idxs = np.random.choice(MEMORY_SIZE, BATCH_SIZE)
        b_memory = self.memory[sample_idxs, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES : N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1 : N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].unsqueeze(1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    dqn = DQN()

    for eps_id in range(100):
        print("========= eps:{}".format(eps_id))
        s, reset_info = env.reset()
        while True:
            env.render()
            a = dqn.choose_action(s)

            s_, r, done, time_up, info = env.step(a)

            # reward shaping
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (
            #     env.theta_threshold_radians - abs(theta)
            # ) / env.theta_threshold_radians - 0.5
            # r = r1 + r2

            dqn.store_eps(s, a, r, s_)

            if dqn.memory_counter > MEMORY_SIZE:
                dqn.learn()

            if done:
                break
            s = s_
    env.close()
