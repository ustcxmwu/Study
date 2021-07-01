import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(object):

    def __init__(self, args, q_trace, learner, actor_critic, rollouts, env_name, actor_name=None):
        self.args = args
        self.q_trace = q_trace
        self.learner = learner
        self.actor_critic = actor_critic
        self.actor_name = actor_name
        self.level = level
        self.env_name = env_name

    def performing(self, rank):
        """
        """
        print('Build Environment for {}'.format(self.actor_name))
        self.env = gym.make(self.env_name)
        torch.manual_seed(self.args.seed)
        writer = SummaryWriter(log_dir=self.args.result_dir)

        self.env.reset()
        state = self.env.observations()
        done = True
        total_reward = 0.
        total_episode_length = 0
        num_episodes = 0

        iterations = 0
        timesteps = 0

        while True:
            self.actor_critic.load_state_dict(self.learner.actor_critic.state_dict())

            for step in range(self.args.num_steps):
                value, action, action_log_prob, recurrent_hidden_states, logits, _ = self.actor_critic.act(
                    self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])
                reward = self.env.step(ACTION_LIST[int(action.item())], num_steps=4)
                state = self.env.observations()
                total_reward += reward

                timesteps += 1
                if done:
                    num_episodes += 1
                    total_episode_length += 1
            self.q_trace.put((
                self.rollouts.obs[:, 0].detach().to("cpu"), self.rollouts.actions[:, 0].detach().to("cpu"),
                self.rollouts.rewards[:, 0].detach().to("cpu"), \
                self.rollouts.action_log_probs[:, 0].detach().to("cpu"),
                self.rollouts.masks[:, 0].detach().to('cpu'),
                self.rollouts.logits[:, 0].detach().to('cpu'),
                self.rollouts.action_onehot[:, 0].detach().to('cpu')))
            if done:
                self.env.reset()
                obs = self.env.observations()['RGB_INTERLEAVED'].transpose((2, 0, 1))
                if timesteps >= self.args.total_num_steps:
                    writer.add_scalar(self.actor_name + '_total_reward', total_reward / num_episodes, iterations)
                    iterations += 1
                    total_reward = 0
                    num_episodes = 0
                    timesteps = 0
