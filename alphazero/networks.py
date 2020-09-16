from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily


class NetworkDiscrete(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, n_hidden_layers: int, n_hidden_units: int
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.in_layer = nn.Linear(self.state_dim, n_hidden_units)

        self.hidden = nn.Sequential(*layers)

        self.policy_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        # no need for softmax, can be computed directly from cross-entropy loss
        pi_logits = self.policy_head(x)
        V_hat = self.value_head(x)
        return pi_logits, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_logits, V_hat = self.forward(states)

        num_actions = actions.shape[1]

        pi_hat = Categorical(
            F.softmax(pi_logits, dim=-1).unsqueeze(dim=0).repeat((num_actions, 1, 1))
        )
        log_probs = pi_hat.log_prob(actions.t()).t()
        entropy = pi_hat.entropy().mean(dim=0)

        return log_probs, entropy, V_hat

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()

    @torch.no_grad()
    def predict_pi(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        pi_hat = F.softmax(self.policy_head(x), dim=-1)
        self.train()
        return pi_hat.detach().cpu().numpy()


class NetworkContinuous(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        act_limit: float,
        n_hidden_layers: int,
        n_hidden_units: int,
        num_components: int,
        log_max: int = 2,
        log_min: int = -5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_components = num_components
        self.act_limit = act_limit

        self.LOG_STD_MAX = log_max
        self.LOG_STD_MIN = log_min

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.in_layer = nn.Linear(self.state_dim, n_hidden_units)

        self.hidden = nn.Sequential(*layers)

        self.mean_head = nn.Linear(n_hidden_units, self.action_dim * num_components)
        self.std_head = nn.Linear(n_hidden_units, self.action_dim * num_components)
        if 1 < self.num_components:
            self.comp_head = nn.Linear(n_hidden_units, num_components)

        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        mean = self.mean_head(x)
        log_std = self.std_head(x)
        V_hat = self.value_head(x)

        # See SpinningUp SAC -> Pre-squashing of distribution
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        if 1 < self.num_components:
            coeffs = torch.softmax(self.comp_head(x), dim=-1)
            return mean, std, coeffs, V_hat
        else:
            return mean, std, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if 1 < self.num_components:
            num_actions = actions.shape[1]
            mean, std, probs, V_hat = self.forward(states)
            coeffs = Categorical(probs.unsqueeze(dim=0).repeat((num_actions, 1, 1)))
            normal = Normal(
                mean.unsqueeze(dim=0).repeat((num_actions, 1, 1)),
                std.unsqueeze(dim=0).repeat((num_actions, 1, 1)),
            )
            gmm = MixtureSameFamily(coeffs, normal)
            log_probs = gmm.log_prob(actions.t()).t()
        else:
            mean, std, V_hat = self.forward(states)
            normal = Normal(mean, std)
            log_probs = normal.log_prob(actions)

        # correct for action bound squashing without summing
        logp_policy = log_probs - (2 * (np.log(2) - actions - F.softplus(-2 * actions)))

        # estimate the entropy for each distribution in the batch
        # the entropy can be estimated using the log_probs as MC samples
        entropy = log_probs.sum(axis=-1)
        entropy -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        return logp_policy, -entropy, V_hat

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        mean = self.mean_head(x)
        log_std = self.std_head(x)
        std = log_std.exp()

        if 1 < self.num_components:
            probs = torch.softmax(self.comp_head(x), dim=-1)
            coeffs = Categorical(probs)
            normal = Normal(mean, std)
            gmm = MixtureSameFamily(coeffs, normal)
            action = gmm.sample()
        else:
            normal = Normal(mean, std)
            action = normal.sample()

        # Enfore action bounds after sampling
        action = torch.tanh(action)
        action = self.act_limit * action
        self.train()

        return action.detach().cpu().numpy()

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()
