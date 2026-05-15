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
        if self.pos + n <= self.max_size:
            (
                self.states[self.pos : self.pos + n],
                self.next_states[self.pos : self.pos + n],
            ) = (states, next_states)
            self.actions[self.pos : self.pos + n] = actions
            (
                self.rewards[self.pos : self.pos + n],
                self.terminations[self.pos : self.pos + n],
            ) = rewards.unsqueeze(1), terminations.unsqueeze(1)
            self.pos += n
            if self.pos == self.max_size:
                self.pos, self.full = 0, True
        else:
            fit, overflow = self.max_size - self.pos, n - (self.max_size - self.pos)
            self.states[self.pos :], self.states[:overflow] = states[:fit], states[fit:]
            self.next_states[self.pos :], self.next_states[:overflow] = (
                next_states[:fit],
                next_states[fit:],
            )
            self.actions[self.pos :], self.actions[:overflow] = (
                actions[:fit],
                actions[fit:],
            )
            self.rewards[self.pos :], self.rewards[:overflow] = rewards[:fit].unsqueeze(
                1
            ), rewards[fit:].unsqueeze(1)
            self.dones[self.pos :], self.dones[:overflow] = terminations[
                :fit
            ].unsqueeze(1), terminations[fit:].unsqueeze(1)
            self.pos, self.full = overflow, True

    def sample(self, batch_size: int):
        idx = torch.randint(
            0,
            self.max_size if self.full else self.pos,
            (batch_size,),
            device=self.device,
        )
        return Batch(
            self.states[idx],
            self.next_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.terminations[idx],
        )
