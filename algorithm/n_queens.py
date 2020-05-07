from copy import deepcopy


def backtrack(board, row, res):
    if row == len(board):
        res.append(deepcopy(board))
        return
    for col in range(len(board)):
        if not is_valid(board, row, col):
            continue
        board[row][col] = "Q"

        backtrack(board, row+1, res)

        board[row][col] = "."


def is_valid(board, row, col):
    for i in range(len(board)):
        if board[i][col] == "Q":
            return False

    # print("row:{}, col:{}".format(row, col))

    # print(list(zip(list(range(row-1, -1, -1)), list(range(col+1, len(board))))))
    for i, j in zip(list(range(row-1, -1, -1)), list(range(col+1, len(board)))):
        if board[i][j] == "Q":
            return False

    # print(list(zip(list(range(row-1, -1, -1)), list(range(col-1, -1, -1)))))
    for i, j in zip(list(range(row-1, -1, -1)), list(range(col-1, -1, -1))):
        if board[i][j] == "Q":
            return False

    return True


if __name__ == '__main__':
    n = 8
    board = [["." for _ in range(n)] for _ in range(n)]
    res = []
    backtrack(board, 0, res)
    print("Total get {} solutions".format(len(res)))
    for i, s in enumerate(res):
        print("the {} th solution in total {} solutions".format(i, len(res)))
        single = "\n".join([" ".join(row) for row in s])
        print(single)
        print()

