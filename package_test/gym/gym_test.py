import gym


if __name__ == '__main__':
    print(len(gym.envs.registry.all()))
    for env in gym.envs.registry.all():
        print(env)


