#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from elo import rate_1vs1 as elo_rate
from trueskill import rate_1vs1 as true_skill_rate


if __name__ == '__main__':
    b = elo_rate(800, 1200)
    print(b)
    c = elo_rate(1200, 800)
    print(c)

    from trueskill import Rating, quality_1vs1, rate_1vs1

    alice, bob = Rating(25), Rating(30)  # assign Alice and Bob's ratings
    if quality_1vs1(alice, bob) < 0.50:
        print('This match seems to be not so fair')
    alice, bob = rate_1vs1(alice, bob)  # update the ratings after the match
    print(alice)
    print(bob)