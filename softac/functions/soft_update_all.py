import torch
from itertools import chain
from typing import List


def soft_update_all(
    *, sources: List[torch.nn.Module], targets: List[torch.nn.Module], tau: float
) -> None:
    for source_parameter, target_parameter in chain(
        *(source.parameters() for source in sources),
        *(target.parameters() for target in targets),
    ):
        target_parameter.data.mul_(1 - tau).add_(source_parameter.data, alpha=tau)
