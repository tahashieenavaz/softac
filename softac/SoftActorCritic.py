import brax.envs
import torch

from collections.abc import ItemsView
from torch.nn.functional import mse_loss
from typing import Any, Tuple, List, Type
from softac.modules import Actor, Critic
from softac.functions import hard_update_all, optimizer_step, soft_update_all
from softac.buffers import Buffer
from brax.envs.wrappers import gym as brax_gym
from brax.envs.wrappers import torch as brax_torch
from baloot import acceleration_device
from baloot import seed as baloot_seed


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
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        actor_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        num_critics: int = 2,
        critic_activation: Type[torch.nn.Module] = torch.nn.GELU,
        actor_activation: Type[torch.nn.Module] = torch.nn.GELU,
        critic_hidden_dimension: int = 256,
        actor_hidden_dimension: int = 256,
        std_min: float = -5,
        std_max: float = 2,
    ):
        self.__set_attributes(locals().items())
        self.device = acceleration_device()
        self.num_updates = self.timesteps // self.num_environments

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

    def __initialize_actor(
        self, state_dimension: int, action_dimension: int, environment
    ) -> Actor:
        return Actor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=self.actor_hidden_dimension,
            high=environment.action_space.high[0],
            low=environment.action_space.low[0],
            std_min=self.std_min,
            std_max=self.std_max,
        ).to(self.device)

    def __initialize_critics(
        self, state_dimension: int, action_dimension: int
    ) -> List[Critic]:
        def __make_critic() -> Critic:
            return Critic(
                state_dimension=state_dimension,
                action_dimension=action_dimension,
                activation=self.critic_activation,
                hidden_dimension=self.critic_hidden_dimension,
            ).to(self.device)

        return [__make_critic() for _ in range(self.num_critics)]

    def __initialize_critic_optimizer(
        self, critics: List[Critic]
    ) -> torch.optim.Optimizer:
        parameters = [param for critic in critics for param in critic.parameters()]
        return torch.optim.Adam(parameters, lr=self.critic_lr)

    def __initialize_actor_optimizer(self, actor: Actor):
        return torch.optim.Adam(list(actor.parameters()), lr=self.actor_lr)

    def __target_entropy(self, action_dimension: int) -> int:
        return -1 * action_dimension

    def __initialize_log_alpha(self) -> torch.Tensor:
        return torch.zeros(1, requires_grad=True, device=self.device)

    def __initialize_log_alpha_optimizer(
        self, log_alpha: torch.Tensor
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam([log_alpha], lr=self.critic_lr)

    def __action_noise(self, action_dimension: int):
        noise = torch.rand((self.num_environments, action_dimension))
        # change the range from [0, 1) to [-1, 1]
        noise = 2 * noise - 1
        return noise

    @torch.inference_mode()
    def __get_actions(
        self, step: int, actor: Actor, states: torch.Tensor, action_dimension: int
    ):
        if step < self.learning_starts:
            action_noise = self.__action_noise(action_dimension=action_dimension)
            return action_noise * actor.scale + actor.bias
        actions, _, _ = actor.get_action(states)
        return actions

    def __get_q(
        self,
        targets: List[Critic],
        next_states: torch.Tensor,
        next_actions: torch.Tensor,
        alpha: torch.Tensor,
        next_log_pi: torch.Tensor,
    ) -> torch.Tensor:
        abel = torch.min(
            targets[0](next_states, next_actions), targets[1](next_states, next_actions)
        )
        cain = alpha * next_log_pi
        return abel - cain

    @torch.inference_mode()
    def __get_targets(
        self,
        actor: Actor,
        targets: List[Critic],
        next_states: torch.Tensor,
        alpha: torch.Tensor,
        terminations: torch.Tensor,
        rewards: torch.Tensor,
    ):
        next_actions, next_log_pi, _ = actor.get_action(next_states)
        q = self.__get_q(
            targets=targets,
            next_actions=next_actions,
            alpha=alpha,
            next_log_pi=next_log_pi,
        )
        return rewards + (1 - terminations) * self.gamma * q

    def __critic_loss(
        self,
        critics: List[Critic],
        q_targets: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        reason = critics[0](states, actions)
        emotion = critics[1](states, actions)
        return mse_loss(reason, q_targets) + mse_loss(emotion, q_targets)

    def __update_critic(
        self,
        *,
        critics: List[Critic],
        q_targets: torch.Tensor,
        actions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        critic_loss = self.__critic_loss(
            critics=critics, q_targets=q_targets, actions=actions
        )
        optimizer_step(optimizer=optimizer, loss=critic_loss)

    def __update_actor(
        self,
        update: int,
        actor: Actor,
        critics: List[Critic],
        states: torch.Tensor,
        alpha: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        if update % self.actor_frequency != 0:
            return

        pi, log_pi, _ = actor.get_action(states)
        actor_loss = (
            (alpha * log_pi) - torch.min(critics[0](states, pi), critics[1](states, pi))
        ).mean()
        optimizer_step(optimizer=optimizer, loss=actor_loss)

    def __get_log_pi(
        self, *, states: torch.Tensor, actor: Actor, gradient: bool = True
    ):
        with torch.set_gradient_enabled(gradient):
            _, log_pi, _ = actor.get_action(states)
        return log_pi

    def __alpha_loss(
        self,
        target_entropy: torch.Tensor,
        log_pi: torch.Tensor,
        log_alpha: torch.Tensor,
    ) -> torch.Tensor:
        return (-log_alpha.exp() * (log_pi + target_entropy)).mean()

    def train(self, seed: int, environment_name: str):
        baloot_seed(seed)
        environment = self.__create_environments(environment_name=environment_name)
        state_dimension, action_dimension = self.__get_environment_properties(
            environment=environment
        )
        actor = self.__initialize_actor(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            environment=environment,
        )
        critics = self.__initialize_critics(
            state_dimension=state_dimension, action_dimension=action_dimension
        )
        targets = self.__initialize_critics()
        log_alpha = self.__initialize_log_alpha()
        hard_update_all(sources=critics, targets=targets)

        critic_optimizer = self.__initialize_critic_optimizer(critics=critics)
        actor_optimizer = self.__initialize_actor_optimizer(actor=actor)
        log_alpha_optimizer = self.__initialize_log_alpha_optimizer(log_alpha=log_alpha)
        target_entropy = self.__target_entropy(action_dimension=action_dimension)

        alpha = log_alpha.exp().item()

        buffer = Buffer(
            max_size=self.buffer_size,
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            device=self.device,
        )

        states, info = environment.reset()

        for update in range(self.num_updates):
            step = update * self.num_environments
            actions = self.__get_actions(
                step=step, actor=actor, action_dimension=action_dimension, states=states
            )
            next_states, rewards, _terminations, _truncations, infos = environment.step(
                actions
            )
            terminations = _terminations.float()
            buffer.add(states, next_states, actions, rewards, terminations)
            states = next_states

            if step > self.learning_starts:
                data = buffer.sample(self.batch_size)
                q_targets = self.__get_targets(
                    actor=actor,
                    targets=targets,
                    alpha=alpha,
                    next_states=data.next_states,
                    terminations=data.terminations,
                )
                self.__update_critic(
                    critics=critics,
                    actions=data.actions,
                    q_targets=q_targets,
                    optimizer=critic_optimizer,
                )
                self.__update_actor(
                    update=update,
                    actor=actor,
                    critics=critics,
                    states=data.states,
                    alpha=alpha,
                    optimizer=actor_optimizer,
                )

                if self.train_alpha:
                    log_pi = self.__get_log_pi(
                        states=data.states, actor=actor, gradient=False
                    )
                    log_pi_loss = self.__alpha_loss(
                        target_entropy=target_entropy,
                        log_pi=log_pi,
                        log_alpha=log_alpha,
                    )
                    optimizer_step(optimizer=log_alpha_optimizer, loss=log_pi_loss)
                    alpha = log_alpha.exp().item()

            if update % self.target_network_frequency == 0:
                soft_update_all(sources=critics, targets=targets)
        pass
