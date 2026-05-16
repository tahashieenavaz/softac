import torch


def soft_update(
    *, source: torch.nn.Module, target: torch.nn.Module, tau: float
) -> None:
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters()
    ):
        target_parameter.data.mul_(1 - tau).add_(source_parameter.data, alpha=tau)
