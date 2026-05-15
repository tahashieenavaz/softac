import torch
from collections import namedtuple

Batch = namedtuple(
    "Batch", ["states", "next_states", "actions", "rewards", "terminations"]
)


class Buffer:
    def __init__(
        self,
        *,
        max_size: int,
        state_dimension: int,
        action_dimension: int,
        device: torch.device
    ):
        self.pointer = 0
        self.full = False
        self.max_size = max_size
        self.device = device

        self.states = torch.zeros((max_size, state_dimension), device=device)
        self.next_states = torch.zeros((max_size, state_dimension), device=device)
        self.actions = torch.zeros((max_size, action_dimension), device=device)
        self.rewards = torch.zeros((max_size, 1), device=device)
        self.terminations = torch.zeros((max_size, 1), device=device)

    def add(self, states, next_states, actions, rewards, terminations):
        n = states.shape[0]

        # we group the data together to avoid multiple repetitive slicing
        data = (
            (self.states, states),
            (self.next_states, next_states),
            (self.actions, actions),
            (self.rewards, rewards.unsqueeze(1)),
            (
                self.terminations,
                terminations.unsqueeze(1),
            ),
        )

        if self.pointer + n <= self.max_size:
            for buf, dat in data:
                buf[self.pointer : self.pointer + n] = dat
            self.pointer += n
            if self.pointer == self.max_size:
                self.pointer = 0
                self.full = True
        else:
            fit, overflow = self.max_size - self.pointer, n - (
                self.max_size - self.pointer
            )
            for buf, dat in data:
                buf[self.pointer :] = dat[:fit]
                buf[:overflow] = dat[fit:]
            self.pointer, self.full = overflow, True

    def sample(self, batch_size: int):
        high = self.max_size if self.full else self.pointer
        idx = torch.randint(0, high, (batch_size,), device=self.device)

        return Batch(
            self.states[idx],
            self.next_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.terminations[idx],
        )
