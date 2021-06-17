#!/usr/bin/python3
from itertools import chain
import json
import os
target = []
visit = None


def get_neighbors(posx, posy, row, col):
    res = []
    for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        if 0 <= posx+x < row and 0 <= posy+y < col:
            res.append((posx+x, posy+y))
    return res


def next_move(posx, posy, board):
    if os.path.exists("./tmp.json"):
        with open("./tmp.json", mode='r') as f:
            tmp = json.load(f)
        visit = tmp["visit"]
        target = tmp["target"]
    else:
        target = []
        visit = [[False for _ in board[0]] for _ in board]
    visit[posx][posy] = True
    if len(target) > 0:
        tx, ty = target[0]
        if tx == posx and ty == posy:
            target.pop(0)
        elif tx < posx:
            print("UP")
        elif tx > posx:
            print("DOWN")
        elif ty < posy:
            print("LEFT")
        else:
            print("RIGHT")
    else:
        neighbors = get_neighbors(posx, posy, len(board), len(board[0]))
        if all([visit[x_][y_] for x_, y_ in neighbors]) and visit[posx][posy]:
            for x in range(len(board)):
                for y in range(len(board[0])):
                    if not visit[x][y]:
                        target.append((x, y))
                        break
                else:
                    continue
                break
        elif board[posx][posy] == "d":
            print("CLEAN")
        else:
            for x, y in neighbors:
                if visit[x][y]:
                    continue
                elif board[x][y] == "-" or board[x][y] == "b":
                    visit[x][y] = True
                elif board[x][y] == "d":
                    target.append((x, y))
        if len(target) > 0:
            tx, ty = target[0]
            if tx == posx and ty == posy:
                target.pop(0)
            elif tx < posx:
                print("UP")
            elif tx > posx:
                print("DOWN")
            elif ty < posy:
                print("LEFT")
            else:
                print("RIGHT")
    with open("./tmp.json", mode='w') as f:
        json.dump({"target": target, "visit": visit}, f)


if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]
    next_move(pos[0], pos[1], board)