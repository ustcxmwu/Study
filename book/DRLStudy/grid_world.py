import sys
from six import StringIO, b
import numpy as np

from gym import utils
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {"4x4": ["SOOO", "OXOX", "OOOX", "XOOG"]}


class HelloGridEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, desc=None, map_name="4x4"):
        self.desc = np.array(MAPS[map_name], dtype="c")
        self.shape = self.desc.shape
        nA = 4
        nS = np.prod(self.desc.shape)
        MAX_X = self.shape[0]
        MAX_Y = self.shape[1]

        isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {}

        state_grid = np.arange(nS).reshape(self.desc.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            x, y = it.multi_index
            P[s] = {a: [] for a in range(nA)}
            s_letter = self.desc[x][y]
            is_done = lambda letter: letter in b'GX'
            reward = 1.0 if s_letter in b'G' else -1.0
            if is_done(s_letter):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if x == 0 else s - MAX_Y
                ns_right = s if y == (MAX_Y - 1) else s + 1
                ns_down = s if x == (MAX_X - 1) else s + MAX_Y
                ns_left = s if y == 0 else s - 1

                s1_up = self.desc[ns_up // MAX_X][ns_up % MAX_Y]
                s1_right = self.desc[ns_right // MAX_X][ns_right % MAX_Y]
                s1_down = self.desc[ns_down // MAX_X][ns_down % MAX_Y]
                s1_left = self.desc[ns_left // MAX_X][ns_left % MAX_Y]

                P[s][UP] = [(1.0, ns_up, reward, is_done(s1_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(s1_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(s1_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(s1_left))]
            it.iternext()

        self.P = P
        super().__init__(nS, nA, P, isd)


    def render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == "ansi" else sys.stdout

        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            x, y = it.multi_index

            if self.s == s:
                desc[x][y] = utils.colorize(desc[x][y], "red", highlight=True)
            it.iternext()
        outfile.write("\n".join(" ".join(line) for line in desc) + "\n")

        if mode != "human":
            return outfile


if __name__ == '__main__':
    env = HelloGridEnv()
    state = env.reset()

    for _ in range(5):
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print("action:{}({})".format(action, ["UP", "RIGHT", "DOWN", "LEFT"][action]))
        print("done:{}, observation:{}, reward:{}".format(done, state, reward))

        if done:
            print("Episode finished after {} timesteps".format(_+1))
            break

