import sys
from collections import defaultdict
import gym

import numpy as np
from utils import plot_value_function


def simple_strategy(state):
    player, dealer, ace = state
    return 0 if player > 18 else 1


def mc_first_visit_prediction(policy, env, num_episodes, episode_end_time=10, discount=1.0):
    r_sum = defaultdict(float)
    r_count = defaultdict(float)
    r_V = defaultdict(float)

    for i in range(num_episodes):
        eposide_rate = int(40*i/num_episodes)
        print("Eposide {}/{}".format(i+1, num_episodes) + "="*eposide_rate, end="\r")
        sys.stdout.flush()

        episode = []
        state = env.reset()

        for j in range(episode_end_time):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward, reward))
            if done:
                break
            state = next_state

        for k, data_k in enumerate(episode):
            state_k = data_k[0]
            G = sum([x[2]*np.power(discount, i) for i, x in enumerate(episode[k:])])
            r_sum[state_k] += G
            r_count[state_k] += 1

            r_V[state_k] = r_sum[state_k] / r_count[state_k]
    return r_V


if __name__ == '__main__':
    env = gym.make("Blackjack-v0")
    v = mc_first_visit_prediction(simple_strategy, env, 10000)
    print(v)
    plot_value_function(v, title=None)