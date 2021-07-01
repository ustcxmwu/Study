#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp


env = gym.make("LunarLander-v2")
low = env.observation_space.low
high = env.observation_space.high


class Critic(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.a = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        a = self.a(x)
        return a


class Agent(object):

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.actor = Actor()
        self.critic = Critic()
        self.log_prob = None

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        # self.gamma = tf.convert_to_tensor(0.99, dtype=tf.double)
        # d = 1 - done
        # d = tf.convert_to_tensor(d, dtype=tf.double)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)

            # p = self.actor(state, training=True).numpy()[0][action]
            # p = tf.convert_to_tensor([[p]], dtype=tf.float32)
            # print(p)
            v = self.critic(state, training=True)
            # v = tf.dtypes.cast(v, tf.double)

            vn = self.critic(next_state, training=True)
            # vn = tf.dtypes.cast(vn, tf.double)
            td = reward + self.gamma * vn * (1 - int(done)) - v
            # print(td)
            # td = tf.math.subtract(tf.math.add(reward, tf.math.multiply(tf.math.multiply(self.gamma, vn), d)), v)
            # a_loss = -self.log_prob*td
            a_loss = self.actor_loss(p, action, td)
            # a_loss = -tf.math.multiply(tf.math.log(p),td)
            # a_loss = tf.keras.losses.categorical_crossentropy(td, p)
            # a_loss = -tf.math.multiply(self.log_prob,td)
            c_loss = td ** 2
            # c_loss = tf.math.pow(td,2)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


if __name__ == '__main__':
    agentoo7 = Agent()
    steps = 100
    for s in range(steps):

        done = False
        state = env.reset()
        total_reward = 0
        all_aloss = []
        all_closs = []

        while not done:
            env.render()
            action = agentoo7.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            aloss, closs = agentoo7.learn(state, action, reward, next_state, done)
            all_aloss.append(aloss)
            all_closs.append(closs)
            state = next_state
            total_reward += reward

            if done:
                # print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(s, total_reward))
