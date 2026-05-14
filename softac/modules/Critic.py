import torch
from typing import Type


class Critic(torch.nn.Module):
    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Linear(observation_dimension + action_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([x, a], 1)
        return self.stream(concatenated)
