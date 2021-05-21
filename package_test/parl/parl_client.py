#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import gym
import numpy as np
import paddle
from parl.utils import logger
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class CartpoleModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(CartpoleModel, self).__init__()
        hid1_size = act_dim * 10
        hid2_size = act_dim * 2
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, x):
        out = paddle.tanh(self.fc1(x))
        out = F.softmax(self.fc2(out), axis=-1)
        prob = F.softmax(self.fc3(out), axis=-1)
        return prob


class CartpoleAgent(parl.Agent):

    def __init__(self, algorithm):
        super(CartpoleAgent, self).__init__(algorithm)

    def sample(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]


LEARNING_RATE = 1e-3


# train an episode
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    env = gym.make('CartPole-v0')
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # build an agent
    model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = parl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
    agent = CartpoleAgent(alg)

    # load model and evaluate
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    for i in range(1000):
        obs_list, action_list, reward_list = run_train_episode(agent, env)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = run_evaluate_episodes(agent, env, render=True)
            logger.info('Test reward: {}'.format(total_reward))

    # save the parameters to ./model.ckpt
    # agent.save('./model.ckpt')


if __name__ == '__main__':
    main()