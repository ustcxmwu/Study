import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from policy_gradient import PolicyGradient


def main():
    np.random.seed(1)
    tf.set_random_seed(1)

    DISPLAY_REWARD_THRESHOLD = 400
    RENDER = False

    env = gym.make("CartPole-v0")

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99,
        output_graph=True
    )

    for i in range(3000):
        observation = env.reset()
        while True:
            if RENDER:
                env.render()

            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)
                if "running_reward" not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward*0.99 + ep_rs_sum*0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print("eposide:{} reward:{}".format(i, int(running_reward)))

                vt = RL.learn()
                if i == 0:
                    plt.plot(vt)
                    plt.xlabel("episode steps")
                    plt.ylabel("normalized state-action value")
                    plt.show()
                break
            observation = observation_


if __name__ == "__main__":
    main()