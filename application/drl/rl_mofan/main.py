from drl.rl_mofan.maze_env import Maze
from drl.rl_mofan.qlearning_sarsa import SarsaLambdaTable
import time


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # RL.learn(str(observation), action, reward, str(observation_))

            # swap observation and action
            observation = observation_
            action = action_
            time.sleep(0.1)

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
