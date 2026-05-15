import brax.envs

from collections.abc import ItemsView
from softac.modules import Actor, Critic
from typing import Any
from typing import Tuple
from brax.envs.wrappers import gym as brax_gym
from brax.envs.wrappers import torch as brax_torch
from baloot import acceleration_device, seed as baloot_seed


class SoftActorCritic:
    def __init__(
        self,
        timesteps: int = 5_000_000,
        num_environments: int = 2048,
        buffer_size: int = 1_000_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 2048,
        learning_starts: int = 10_000,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
    ):
        self.__set_attributes(locals().items())
        self.device = acceleration_device()

    def __set_attributes(self, items: ItemsView[str, Any]) -> None:
        for key, value in items:
            if key == "self":
                continue
            setattr(self, key, value)

    def __create_environments(self, *, environment_name: str):
        env = brax.envs.create(
            environment_name, batch_size=self.num_environments, backend="spring"
        )
        env = brax_gym.VectorGymWrapper(env)
        env = brax_torch.TorchWrapper(env, device=self.device)
        return env

    def __get_environment_properties(self, environment) -> Tuple[int, int]:
        state_dimension = environment.observation_space.shape[-1]
        action_dimension = environment.action_space.shape[-1]
        return state_dimension, action_dimension
    
    def __initialize_actor(self, state_dimension: int, action_dimension: int, environment):
        return Actor(state_dimension, action_dimension, env, self.device).to(self.device)

    def train(self, seed: int, environment_name: str):
        baloot_seed(seed)
        environment = self.__create_environments(environment_name=environment_name)
        state_dimension, action_dimension = self.__get_environment_properties(
            environment=environment
        )
        actor = 
        critics = [
            Critic(obs_dim, act_dim).to(device),
            Critic(obs_dim, act_dim).to(device),
        ]
        targets = [
            Critic(obs_dim, act_dim).to(device),
            Critic(obs_dim, act_dim).to(device),
        ]
        for index, critic in enumerate(critics):
            targets[index].load_state_dict(critic.state_dict())

        pass
