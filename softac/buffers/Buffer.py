import torch


class Buffer:
    def __init__(self, max_size: int, obs_dim: int, act_dim: int, device):
        self.pos, self.full, self.max_size, self.device = 0, False, max_size, device
        self.obs = torch.zeros((max_size, obs_dim), device=device)
        self.next_obs = torch.zeros((max_size, obs_dim), device=device)
        self.actions = torch.zeros((max_size, act_dim), device=device)
        self.rewards = torch.zeros((max_size, 1), device=device)
        self.dones = torch.zeros((max_size, 1), device=device)

    def add(self, obs, next_obs, actions, rewards, dones):
        n = obs.shape[0]
        if self.pos + n <= self.max_size:
            (
                self.obs[self.pos : self.pos + n],
                self.next_obs[self.pos : self.pos + n],
            ) = (obs, next_obs)
            self.actions[self.pos : self.pos + n] = actions
            (
                self.rewards[self.pos : self.pos + n],
                self.dones[self.pos : self.pos + n],
            ) = rewards.unsqueeze(1), dones.unsqueeze(1)
            self.pos += n
            if self.pos == self.max_size:
                self.pos, self.full = 0, True
        else:
            fit, overflow = self.max_size - self.pos, n - (self.max_size - self.pos)
            self.obs[self.pos :], self.obs[:overflow] = obs[:fit], obs[fit:]
            self.next_obs[self.pos :], self.next_obs[:overflow] = (
                next_obs[:fit],
                next_obs[fit:],
            )
            self.actions[self.pos :], self.actions[:overflow] = (
                actions[:fit],
                actions[fit:],
            )
            self.rewards[self.pos :], self.rewards[:overflow] = rewards[:fit].unsqueeze(
                1
            ), rewards[fit:].unsqueeze(1)
            self.dones[self.pos :], self.dones[:overflow] = dones[:fit].unsqueeze(
                1
            ), dones[fit:].unsqueeze(1)
            self.pos, self.full = overflow, True

    def sample(self, batch_size):
        idx = torch.randint(
            0,
            self.max_size if self.full else self.pos,
            (batch_size,),
            device=self.device,
        )
        return Batch(
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )
