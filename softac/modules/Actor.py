import torch


class Actor(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        high: int,
        low: int,
        activation=torch.nn.ReLU,
        device
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            activation(),
        )
        self.mu = torch.nn.Linear(hidden_dimension, action_dimension)
        self.sigma = torch.nn.Linear(hidden_dimension, action_dimension)
        # env.actionspace.high
        high = torch.tensor(high, dtype=torch.float32, device=device)[0]
        low = torch.tensor(low, dtype=torch.float32, device=device)[0]
        self.scale = (high - low) / 2.0
        self.bias = (high + low) / 2.0

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        mean = self.mu(x)
        std = -5 + 0.5 * (2 - (-5)) * (torch.tanh(self.sigma(x)) + 1)
        normal = torch.distributions.Normal(mean, std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(
            self.scale * (1 - y_t.pow(2)) + 1e-6
        )
        return (
            y_t * self.scale + self.bias,
            log_prob.sum(1, keepdim=True),
            torch.tanh(mean) * self.scale + self.bias,
        )
