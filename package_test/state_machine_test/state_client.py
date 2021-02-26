#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from statemachine import StateMachine, State


class Life(StateMachine):
class Life(object):
    "A traffic light machine"
    workday = State('Workday', initial=True)
    weekend = State('Weekend')

    cycle = workday.to(weekend) | weekend.to(workday)

    def on_enter_workday(self):
        print('Start work')

    def on_enter_weekend(self):
        print('Have a rest')


if __name__ == '__main__':
    life = Life()
    for i in range(100):
        life.cycle()