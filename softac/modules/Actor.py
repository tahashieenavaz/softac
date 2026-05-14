import torch


class Actor(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, env: int, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.fc_mean, self.fc_logstd = torch.nn.Linear(256, act_dim), torch.nn.Linear(
            256, act_dim
        )

        high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)[
            0
        ]
        low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)[0]
        self.scale, self.bias = (high - low) / 2.0, (high + low) / 2.0

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        mean, log_std = self.fc_mean(x), -5 + 0.5 * (2 - (-5)) * (
            torch.tanh(self.fc_logstd(x)) + 1
        )
        normal = torch.distributions.Normal(mean, log_std.exp())
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
