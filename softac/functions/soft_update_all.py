import torch
from typing import List
from .soft_update import soft_update


def soft_update_all(
    *, sources: List[torch.nn.Module], targets: List[torch.nn.Module], tau: float
) -> None:
    for source, target in zip(sources, targets):
        soft_update(source=source, target=target, tau=tau)
