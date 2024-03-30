import gym

if __name__=='__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
