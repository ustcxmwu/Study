import numpy as np
from agent import ModelFreeAgent, eval_game
from snake import SnakeEnv
from utils import timer


class MonteCarlo(object):

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def monte_carlo_opt(self, agent, env):
        for i in range(10):
            for j in range(100):
                self.monte_carlo_eval(agent, env)
            self.policy_improve(agent)

    def monte_carlo_eval(self, agent, env):
        state = env.reset()
        episode = []
        while True:
            ac = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(ac)
            episode.append((state, ac, reward))
            state = next_state
            if terminate:
                break
        value = []
        return_val = 0
        for _state, _a, _reward in reversed(episode):
            return_val = return_val * agent.gamma + _reward
            value.append((_state, _a, return_val))

        for _s, _a, _r in reversed(value):
            agent.value_n[_s][_a] += 1
            agent.value_q[_s][_a] += (_r - agent.value_q[_s][_a]) / agent.value_n[_s][_a]

    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True


def monte_carlo_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3, 6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo()
    with timer("MonteCarlo Iter"):
        mc.monte_carlo_opt(agent, env)
    print("return_pi={}".format(eval_game(env, agent)))
    print(agent.pi)


if __name__ == '__main__':
    monte_carlo_demo()
