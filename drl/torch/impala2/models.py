import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size=256):
        super().__init__()
        self.n_actions = action_space.n
        self.main = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.main(inputs)
        return self.critic_linear(x)

    def act(self, inputs):
        prob = self.main(inputs)
        m = Categorical(prob)
        action = m.sample().item()
        value = self.forward(inputs)
        return value, action, prob

    def get_value(self, inputs):
        value = self.main(inputs)
        return value


    def evaluate_action(self):
        pass

