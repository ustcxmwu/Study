import numpy as np
from grid_world import HelloGridEnv


def calc_action_value(env, state, V, discount_factor=1.0):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


def value_iteration(env, theta=0.1, discount_factor=1.0):
    V = np.zeros(env.nS)
    for _ in range(50):
        delta = 0
        for s in range(env.nS):
            A = calc_action_value(env, s, V)
            best_action_value = np.max(A)

            delta = max(delta, np.abs(best_action_value - V[s]))

            V[s] = best_action_value
        if delta < theta:
            break
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = calc_action_value(env, s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy, V


def main():
    env = HelloGridEnv()
    policy, v = value_iteration(env)

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))

    print("Reshaped Grid Policy(0: up, 1: right, 2: down, 3: left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))


if __name__ == '__main__':
    main()