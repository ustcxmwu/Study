import numpy as np


class TableAgent(object):

    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.r = [env.rewards(s) for s in range(0, self.s_len)]
        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, 100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step > 100, step <= 100], [lambda x: 200 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob
        self.p[:, 100, 100] = 1
        self.value_pi = np.zeros((self.s_len))
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]


class ModelFreeAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.value_n = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return self.pi[state]


def eval_game(env, policy):
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy, TableAgent) or isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise ValueError("Illegal policy")
        state, reward, terminate, _ = env.step(act)
        return_val += reward
        if terminate:
            break
    return return_val


def test_play():
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0
    import snake
    snake = snake.SnakeEnv(0, [3, 6])
    policy_ref = [1] * 97 + [0] * 3
    policy_0 = [0] * 100
    policy_1 = [1] * 100
    for i in range(10000):
        sum_opt += eval_game(snake, policy_ref)
        sum_0 += eval_game(snake, policy_0)
        sum_1 += eval_game(snake, policy_1)
    print("opt avg:{}".format(sum_opt / 10000.0))
    print("0 avg={}".format(sum_0 / 10000.0))
    print("1 avg={}".format(sum_1 / 10000.0))


if __name__ == '__main__':
    test_play()
