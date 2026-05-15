import torch
from typing import List
from .hard_update import hard_update


def hard_update_all(
    sources: List[torch.nn.Module], targets: List[torch.nn.Module]
) -> None:
    assert len(sources) == len(targets), "Sources and targets must have same length."
    for source, target in zip(sources, targets):
        hard_update(source=source, target=target)
