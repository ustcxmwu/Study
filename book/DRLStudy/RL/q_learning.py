import numpy as np
from snake import SnakeEnv
from agent import ModelFreeAgent, eval_game
from utils import timer


class QLearning(object):

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def q_learning_eval(self, agent, env):
        episode_num = 1000
        for i in range(episode_num):
            state = env.reset()
            prev_act = -1
            prev_state = -1
            while True:
                act = agent.play(state, self.epsilon)
                next_state, reward, terminate, _ = env.step(act)
                if prev_act != -1:
                    return_val = reward + agent.gamma * (0 if terminate else np.max(agent.value_q[state, :]))
                    agent.value_n[prev_state][prev_act] += 1
                    agent.value_q[prev_state][prev_act] += (return_val - agent.value_q[prev_state][prev_act]) / \
                                                           agent.value_n[prev_state][prev_act]
                prev_act = act
                prev_state = state
                state = next_state

                if terminate:
                    break

    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def q_learning(self, agent, env):
        iteration = 0
        while True:
            iteration += 1
            self.q_learning_eval(agent, env)
            ret = self.policy_improve(agent)
            if not ret:
                break


def q_learning_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3, 6])
    agent = ModelFreeAgent(env)
    mc = QLearning(0.5)
    with timer("SARSA Iter"):
        mc.q_learning(agent, env)
    print("return_pi={}".format(eval_game(env, agent)))
    print(agent.pi)


if __name__ == '__main__':
    q_learning_demo()
