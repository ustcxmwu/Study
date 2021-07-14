import numpy as np
import gym

from gym.spaces import Discrete


class SnakeEnv(gym.Env):
    SIZE = 100

    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(dices))

        back_ladders = dict()
        for k, v in self.ladders.items():
            back_ladders[v] = k

        self.ladders.update(back_ladders)
        print("ladders info:")
        print(self.ladders)
        print("dice ranges:")
        print(self.dices)

        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, action):
        step = np.random.randint(1, self.dices[action] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def rewards(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    env = SnakeEnv(10, [3, 6])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(0)
        print("state:{}, reward:{}".format(state, reward))
        if terminate == 1:
            break


