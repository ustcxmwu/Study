import gym
from gym import utils


def simple_strategy(state):
    player, dealer, ace = state
    return 0 if player > 18 else 1


env = gym.make("Blackjack-v0")


def show_state(state):
    player, dealer, ace = state
    dealer = sum(env.dealer)
    print("Player:{}, ace:{}, Dealer:{}".format(player, ace, dealer))


def episode(num_episodes):
    episode = []
    for i in range(num_episodes):
        print("\n" + "=" * 30)
        state = env.reset()
        for t in range(10):
            show_state(state)
            action = simple_strategy(state)
            action_ = ["STAND", "HIT"][action]
            print("Player take action:{}".format(action_))

            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                show_state(state)
                reward_ = ["loss", "push", "win"][int(reward)]
                print("Game {}.(Reward {})".format(reward_, int(reward)))
                print("Player:{}\t Dealer:{}\t".format(utils.colorize(env.player, "red"), utils.colorize(env.dealer, "green")))
                break
            state = next_state


if __name__ == '__main__':
    episode(1000)
