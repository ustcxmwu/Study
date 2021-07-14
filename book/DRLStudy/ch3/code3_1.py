from grid_world import HelloGridEnv
import numpy as np


def policy_evaluation(policy, environment, discount_factor=1.0, theta=0.1):
    env = environment
    V = np.zeros(env.nS)
    for _ in range(10000):
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor*V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta <= theta:
            break
    return np.array(V)


def policy_iteration(env, policy, discount_factor=1.0):
    while True:
        V = policy_evaluation(policy, env, discount_factor)
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor*V[next_state])
                    if done and next_state != 15:
                        action_values[a] = float("-inf")

            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_action]

        if policy_stable:
            return policy, V


def main():
    env = HelloGridEnv()
    random_policy = np.ones((env.nS, env.nA))/env.nA
    policy, v = policy_iteration(env, random_policy)

    print("\nReshaped Grid Policy(0:up, 1:right, 2:down, 3:left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("Reshaped Grid Value function:")
    print(v.reshape(env.shape))



if __name__ == '__main__':
    # env = HelloGridEnv()
    # random_policy = np.ones([env.nS, env.nA])/ env.nA
    # print(random_policy)
    #
    # v = policy_evaluation(random_policy, env)
    # print("Reshaped Grid Value Function:")
    # print(v.reshape(env.shape))

    main()
