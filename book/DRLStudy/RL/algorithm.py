import numpy as np

from agent import TableAgent, eval_game
from snake import SnakeEnv
from utils import timer


class PolicyIteration(object):

    def policy_evaluation(self, agent, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = agent.value_pi.copy()
            for i in range(1, agent.s_len):
                ac = agent.pi[i]
                for j in range(0, agent.a_len):
                    transition = agent.p[ac, i, :]
                    value_sa = np.dot(transition, agent.r + agent.gamma*agent.value_pi)
                new_value_pi[i] = value_sa
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break
        print("policy iteration proceed {} iters".format(iteration))

    def policy_improvement(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r  + agent.gamma*agent.value_pi)
            max_act = np.argmax(agent.value_q[i, :])
            new_policy[i] = max_act
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def policy_iteration(self, agent):
        iteration = 0
        while True:
            iteration += 1
            self.policy_evaluation(agent)
            ret = self.policy_improvement(agent)
            if not ret:
                break
        print("Iter {} rounds converge".format(iteration))


class TimerPolicyIteration(PolicyIteration):

    def policy_iteration(self, agent, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            with timer("Timer Policy Evaluation"):
                self.policy_evaluation(agent, max_iter)
            with timer("Timer Policy Improve"):
                ret = self.policy_improvement(agent)
            if not ret:
                break


class ValueIteration(object):

    def value_iteration(self, agent, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = np.zeros_like(agent.value_pi)
            for i in range(1, agent.s_len):
                value_sas = []
                for j in range(0, agent.a_len):
                    value_sa = np.dot(agent.p[j, i, :], agent.r + agent.gamma*agent.value_pi)
                    value_sas.append(value_sa)
                new_value_pi[i] = max(value_sas)
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break
        print("Iter {} rounds converge".format(iteration))
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma*agent.value_pi)
            max_act = np.argmax(agent.value_q[i,:])
            agent.pi[i] = max_act



def demo1():
    env = SnakeEnv(0, [3, 6])
    agent = TableAgent(env)
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print("return pi: {}".format(eval_game(env, agent)))
    print(agent.pi)


def demo2():
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    agent.pi[:] = 0
    print("0 avg={}".format(eval_game(env, agent)))
    agent.pi[:] = 1
    print("1 avg={}".format(eval_game(env, agent)))
    agent.pi[97:100] = 0
    print("opt avg={}".format(eval_game(env, agent)))
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print("pi avg={}".format(eval_game(env, agent)))
    print(agent.pi)


def timer_demo1():
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    pi_algo = TimerPolicyIteration()
    with timer("Policy Iteration"):
        pi_algo.policy_iteration(agent)
    print("return_pi: {}".format(eval_game(env, agent)))
    print(agent.pi)


def timer_demo2():
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    vi_algo = ValueIteration()
    with timer("Value Iteration"):
        vi_algo.value_iteration(agent)
    print("return_pi: {}".format(eval_game(env, agent)))
    print(agent.pi)


if __name__ == '__main__':
    timer_demo1()
    timer_demo2()

