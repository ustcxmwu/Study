import numpy as np

if __name__ == '__main__':
    print('test')
    a = [1, 2, 3, 4]
    for i, val in enumerate(a):
        print(i, val)

    print("a.{}.b".format('wo'))
    print(np.pi/6*np.arange(0, 12))