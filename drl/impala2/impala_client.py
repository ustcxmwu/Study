import argparse
import os
import shutil

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from actor import Actor
from learner import Learner
from models import Policy
from q_manager import QManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', type=float, default=0.00048, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of Steps to learn')
    parser.add_argument('--total_num_steps', type=int, default=4096, help='Number of Steps to learn')
    parser.add_argument('--seed', type=int, default=2019, help='Random seed')
    parser.add_argument('--coef_hat', type=float, default=1.0)
    parser.add_argument('--rho_hat', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount rate')
    parser.add_argument('--entropy_coef', type=float, default=0.00025)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=40)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--reward_clipping', type=str, default='abs_one', choices=['abs_one', 'soft_asymmetric'])
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    processes = []
    q_trace = Queue(maxsize=300)
    q_batch = Queue(maxsize=3)
    q_manager = QManager(args, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()
    processes.append(p)

    envs = []
    actors = []

    args.result_dir = os.path.join('results', str(args.experiment_id))
    args.model_dir = os.path.join(args.result_dir, 'models')

    try:
        os.makedirs(args.model_dir)
    except:
        shutil.rmtree(args.model_dir)
        os.makedirs(args.model_dir)

    env = gym.make("CartPole-v1")

    print('Observation Space: ', env.observation_space)

    actor_critic = Policy(env.observation_space, env.action_space)
    actor_critic.to(args.device)

    learner = Learner(args, q_batch, actor_critic)

    for i in range(3):
        print('Build Actor {:d}'.format(i))
        actor_critic = Policy(env.observation_space, env.action_space)
        actor_critic.to(args.device)

        actor_name = 'actor_' + str(i)
        actor = Actor(args, q_trace, learner, actor_critic, rollouts, LEVELS[i], actor_name)
        actors.append(actor)

    print('Run processes')

    for rank, a in enumerate(actors):
        p = mp.Process(target=a.performing, args=(rank,))
        p.start()
        processes.append(p)

    learner.learning()

    for p in processes:
        p.join()
