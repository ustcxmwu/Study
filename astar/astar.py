import numpy
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Tuple, List, Dict


def heuristic_cost_estimate(neighbor: List[int], goal: List[int]):
    x = neighbor[0] - goal[0]
    y = neighbor[1] - goal[1]
    return abs(x) + abs(y)


def dist_between(a: List[int], b: List[int]):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def reconstruct_path(came_from: Dict, current: List):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# astar function returns a list of points (shortest path)
def astar(array, start: Tuple, goal: Tuple):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8个方向

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic_cost_estimate(start, goal)}

    open_set = []
    heappush(open_set, (fscore[start], start))  # 往堆中插入一条新的值

    # while open_set is not empty
    while open_set:
        # current := the node in open_set having the lowest fScore value
        current = heappop(open_set)[1]  # 从堆中弹出fscore最小的节点

        if current == goal:
            return reconstruct_path(came_from, current)

        close_set.add(current)

        for i, j in directions:  # 对当前节点的 8 个相邻节点一一进行检查
            neighbor = current[0] + i, current[1] + j

            ## 判断节点是否在地图范围内，并判断是否为障碍物
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:  # 1为障碍物
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            # Ignore the neighbor which is already evaluated.
            if neighbor in close_set:
                continue

            #  The distance from start to a neighbor via current
            tentative_gScore = gscore[current] + dist_between(current, neighbor)

            if neighbor not in [i[1] for i in open_set]:  # Discover a new node
                heappush(open_set, (fscore.get(neighbor, numpy.inf), neighbor))
            elif tentative_gScore >= gscore.get(neighbor, numpy.inf):  # This is not a better path.
                continue

                # This path is the best until now. Record it!
            came_from[neighbor] = current
            gscore[neighbor] = tentative_gScore
            fscore[neighbor] = tentative_gScore + heuristic_cost_estimate(neighbor, goal)

    return False


if __name__ == "__main__":
    nmap = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    fig1 = plt.figure(1)
    cmap1 = colors.ListedColormap(['white', 'red'])
    bounds1 = [0, 0.5, 1]
    norm1 = colors.BoundaryNorm(bounds1, cmap1.N)
    plt.imshow(nmap, cmap=cmap1, norm=norm1)

    path = astar(nmap, (0, 3), (10, 1))

    for i in range(len(path)):
        nmap[path[i]] = 100

    fig2 = plt.figure(2)
    cmap2 = colors.ListedColormap(['white', 'red', 'blue'])
    bounds2 = [0, 0.5, 1.5, 100]
    norm2 = colors.BoundaryNorm(bounds2, cmap2.N)
    plt.imshow(nmap, cmap=cmap2, norm=norm2)

    plt.show()
