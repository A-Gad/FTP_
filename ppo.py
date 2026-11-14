from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 64
    hidden_sizes: tuple[int, ...] = (256, 256)
    action_std: float = 1.0
    initial_action: tuple[float, ...] = ()


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        hidden_sizes: tuple[int, ...] = (256, 256),
        action_std: float = 1.0,
        initial_action: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_std <= 0:
            raise ValueError("action_std must be positive")

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim != 1 or high.shape != low.shape:
            raise ValueError("action bounds must be one-dimensional and have equal shape")
        if torch.any(high <= low):
            raise ValueError("every action high bound must exceed its low bound")

        def build_backbone():
            layers: list[nn.Module] = []
            input_dim = observation_dim
            for hidden_size in hidden_sizes:
                if hidden_size <= 0:
                    raise ValueError("hidden layer sizes must be positive")
                layers.extend((nn.Linear(input_dim, hidden_size), nn.Tanh()))
                input_dim = hidden_size
            return nn.Sequential(*layers)

        self.actor_backbone = build_backbone()
        self.critic_backbone = build_backbone()
        input_dim = hidden_sizes[-1] if hidden_sizes else observation_dim
        self.actor_mean = nn.Linear(input_dim, low.numel())
        self.critic_value = nn.Linear(input_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((low.numel(),), np.log(action_std), dtype=torch.float32)
        )
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)

        self._initialize_weights()
        if initial_action is not None:
            initial = torch.as_tensor(initial_action, dtype=torch.float32)
            if initial.shape != low.shape:
                raise ValueError("initial_action must match the action shape")
            normalized = ((initial - self.action_bias) / self.action_scale).clamp(
                -1 + 1e-6, 1 - 1e-6
            )
            self.actor_mean.bias.data.copy_(torch.atanh(normalized))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)

    def _features(self, observations: torch.Tensor):
        return self.actor_backbone(observations), self.critic_backbone(observations)

    def _distribution(self, observations: torch.Tensor):
        actor_features, critic_features = self._features(observations)
        mean = self.actor_mean(actor_features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), critic_features

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        _, critic_features = self._distribution(observations)
        return self.critic_value(critic_features).squeeze(-1)

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(raw_action)

    def _unsquash(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (action - self.action_bias) / self.action_scale
        clipped = normalized.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_action = torch.atanh(clipped)
        correction = torch.log(self.action_scale) + torch.log1p(-clipped.square())
        return raw_action, correction.sum(dim=-1)

    def sample_action(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, critic_features = self._distribution(observations)
        raw_action = distribution.rsample()
        action = self._squash(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(self.action_scale).sum()
        log_prob -= torch.log1p(-torch.tanh(raw_action).square()).sum(dim=-1)
        value = self.critic_value(critic_features).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, critic_features = self._distribution(observations)
        raw_action, correction = self._unsquash(actions)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1) - correction
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic_value(critic_features).squeeze(-1)
        return log_prob, entropy, value

    def deterministic_action(self, observations: torch.Tensor) -> torch.Tensor:
        distribution, _ = self._distribution(observations)
        return self._squash(distribution.mean)


class RolloutBuffer:
    def __init__(self, size: int, observation_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        if size <= 0:
            raise ValueError("rollout size must be positive")
        self.size = size
        self.observations = np.zeros((size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.terminated = np.zeros(size, dtype=np.bool_)
        self.truncated = np.zeros(size, dtype=np.bool_)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        # Store the value of the final observation for truncated episodes
        # so bootstrapping uses the correct state, not the next episode's first state.
        self.truncated_values = np.zeros(size, dtype=np.float32)
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        value: float,
        log_prob: float,
        truncated_value: float = 0.0,
    ):
        if self.position >= self.size:
            raise RuntimeError("rollout buffer is full")
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.terminated[self.position] = terminated
        self.truncated[self.position] = truncated
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.truncated_values[self.position] = truncated_value
        self.position += 1

    def finish(self, last_value: float, gamma: float, gae_lambda: float):
        if self.position != self.size:
            raise RuntimeError("rollout buffer must be full before finish")
        advantage = 0.0
        for index in reversed(range(self.size)):
            if self.truncated[index]:
                # Use the stored value of the truncated episode's final obs,
                # not the next episode's first-state value.
                next_value = self.truncated_values[index]
            elif index == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[index + 1]
            bootstrap = 0.0 if self.terminated[index] else 1.0
            delta = self.rewards[index] + gamma * next_value * bootstrap - self.values[index]
            boundary = self.terminated[index] or self.truncated[index]
            advantage = delta + gamma * gae_lambda * (0.0 if boundary else advantage)
            self.advantages[index] = advantage
        self.returns = self.advantages + self.values

    def tensors(self, device: torch.device):
        if self.position != self.size:
            raise RuntimeError("rollout buffer is not full")
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return (
            torch.as_tensor(self.observations, device=device),
            torch.as_tensor(self.actions, device=device),
            torch.as_tensor(self.log_probs, device=device),
            torch.as_tensor(self.returns, device=device),
            advantages,
        )


class PPO:
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.policy = ActorCritic(
            observation_dim,
            action_low,
            action_high,
            hidden_sizes=self.config.hidden_sizes,
            action_std=self.config.action_std,
            initial_action=(
                self.config.initial_action if self.config.initial_action else None
            ),
        ).to(self.device)
        # Separate optimizers for actor and critic to prevent value-loss
        # gradients from drowning the actor's gradient under shared grad clip.
        actor_params = list(self.policy.actor_backbone.parameters()) + \
                       list(self.policy.actor_mean.parameters()) + \
                       [self.policy.log_std]
        critic_params = list(self.policy.critic_backbone.parameters()) + \
                        list(self.policy.critic_value.parameters())
        self.optimizer_actor = torch.optim.Adam(
            actor_params, lr=self.config.learning_rate, eps=1e-5
        )
        self.optimizer_critic = torch.optim.Adam(
            critic_params, lr=self.config.learning_rate, eps=1e-5
        )

    def act(self, observation: np.ndarray, deterministic: bool = False):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic_action(observation_tensor)
                value = self.policy.value(observation_tensor)
                return action.squeeze(0).cpu().numpy(), None, float(value.item())
            action, log_prob, value = self.policy.sample_action(observation_tensor)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            if deterministic:
                actions = self.policy.deterministic_action(observation_tensor)
                values = self.policy.value(observation_tensor)
                return actions.cpu().numpy(), None, values.cpu().numpy()
            actions, log_probs, values = self.policy.sample_action(observation_tensor)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        observations, actions, old_log_probs, returns, advantages = buffer.tensors(
            self.device
        )
        sample_count = observations.shape[0]
        if self.config.minibatch_size > sample_count:
            raise ValueError("minibatch_size cannot exceed rollout size")

        metrics: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []
        }
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                log_probs, entropy, values = self.policy.evaluate_actions(
                    observations[indices], actions[indices]
                )
                log_ratio = log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                surrogate_first = ratio * advantages[indices]
                surrogate_second = torch.clamp(
                    ratio,
                    1 - self.config.clip_coef,
                    1 + self.config.clip_coef,
                ) * advantages[indices]
                policy_loss = -torch.minimum(surrogate_first, surrogate_second).mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy_mean

                value_loss = 0.5 * (returns[indices] - values).square().mean()

                # Update actor with its own optimizer and grad clip
                self.optimizer_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_params = list(self.policy.actor_backbone.parameters()) + \
                               list(self.policy.actor_mean.parameters()) + \
                               [self.policy.log_std]
                nn.utils.clip_grad_norm_(actor_params, self.config.max_grad_norm)
                self.optimizer_actor.step()

                # Update critic with its own optimizer and grad clip
                self.optimizer_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                critic_params = list(self.policy.critic_backbone.parameters()) + \
                                list(self.policy.critic_value.parameters())
                nn.utils.clip_grad_norm_(critic_params, self.config.max_grad_norm)
                self.optimizer_critic.step()

                with torch.no_grad():
                    metrics["policy_loss"].append(float(policy_loss.item()))
                    metrics["value_loss"].append(float(value_loss.item()))
                    metrics["entropy"].append(float(entropy_mean.item()))
                    metrics["approx_kl"].append(float((old_log_probs[indices] - log_probs).mean().item()))
                    metrics["clip_fraction"].append(float((torch.abs(ratio - 1) > self.config.clip_coef).float().mean().item()))

        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.policy.load_state_dict(state["policy"])
        if "optimizer_actor" in state:
            self.optimizer_actor.load_state_dict(state["optimizer_actor"])
            self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        elif "optimizer" in state:
            pass  # Old checkpoint format — skip optimizer restore