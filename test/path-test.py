import random


def wander():
    npc_x_velocity = random.randint(-5, 5)
    npc_y_velocity = random.randint(-5, 5)
    npc_move_count = 0
    npc_x = 0
    npc_y = 0
    while npc_move_count < 100:
        npc_x += npc_x_velocity
        npc_y += npc_y_velocity
        npc_move_count += 1
        print(npc_x, npc_y)


if __name__ == '__main__':
    wander()