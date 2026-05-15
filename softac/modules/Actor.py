import torch
from torch.distributions import Normal
from typing import Type


class Actor(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        high: float,
        low: float,
        std_max: float,
        std_min: float,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            activation(),
        )
        self.mu = torch.nn.Linear(hidden_dimension, action_dimension)
        self.sigma = torch.nn.Linear(hidden_dimension, action_dimension)
        self.register_buffer(
            "scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
        )

        self.sigma_range_scale = (std_max - std_min) / 2.0
        self.sigma_range_bias = (std_max + std_min) / 2.0
        self.epsilon = epsilon

    def get_std(self, features: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma(features)
        activated_sigma = torch.tanh(sigma)
        return self.sigma_range_scale * activated_sigma + self.sigma_range_bias

    def get_action(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(states)
        mean = self.mu(features)
        std = self.get_std(features=features)
        distribution = Normal(mean, std.exp())
        x_t = distribution.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        mean_action = self.scale * torch.tanh(mean) + self.bias
        log_prob = distribution.log_prob(x_t) - torch.log(
            self.scale * (1 - y_t.pow(2)) + self.epsilon
        )
        return action, log_prob.sum(1, keepdim=True), mean_action
