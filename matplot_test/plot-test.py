import matplotlib.pyplot as plt
import numpy as np


def plot():
    lines = np.array([
        [180, 120],
        [320, 120],
        [380, 180],
        [380, 320],
        [320, 380],
        [180, 380],
        [120, 320],
        [120, 180],
    ])

    for i in range(len(lines)):
        # plt.plot(x[i], y[i], color='r')
        plt.scatter(lines[i][0], lines[i][1], color='b')
        plt.plot([lines[i][0], lines[(i+1) % len(lines)][0]], [lines[i][1], lines[(i+1) % len(lines)][1]], color='r')
    plt.xlim((0, 500))
    plt.ylim((0, 500))

    plt.scatter(450, 285, color='g')
    plt.scatter(300, 300, color='g')
    # plt.scatter(297, 262, color='g')
    # plt.scatter(443, 245, color='g')
    plt.show()


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(np.dot(a, b))
    print(np.multiply(a, b))
    print(np.cross(a, b))
    print(min(3, 4))
    print(max(3, 4))
    c = [a[1], a[2]]
    a[2] = 5
    print(c)

# Create the frames
frames = []
x, y = 0, 0
for i in range(10):
    new_frame = create_image_with_ball(400, 400, x, y, 40)
    frames.append(new_frame)
    x += 40
    y += 40

# Save into a GIF file that loops forever
frames[0].save('moving_ball.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
