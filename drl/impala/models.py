import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class MlpPolicy(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ).to(device).to(dtype)
        )

    def forward(self, state):
        logits = self.model(state)
        return logits

    def select_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        if deterministic:
            action = torch.argmax(logits)
        else:
            action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        return action, logits


class MlpValueFn(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
                .to(device)
                .to(dtype)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)
