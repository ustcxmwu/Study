import imageio
import numpy as np


if __name__ == '__main__':
    # gif = imageio.mimread('1.gif')
    # print(len(gif))
    # print('_'.join(str(x) for x in np.arange(-2, 3)))
    a = np.array([
        [180, 120],
        [320, 120]
    ])
    b = a
    b[0, 1] = 500

    for id in range(len(a)):
        print(np.linalg.norm(a[id] - a[(id+1)%len(a)]))

    print(100<float('inf'))
