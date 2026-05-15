import torch
from typing import Type


class Critic(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.logic = torch.nn.Sequential(
            torch.nn.Linear(state_dimension + action_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([states, actions], 1)
        return self.logic(concatenated)
