import torch
from torch import Tensor
from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod


from diffusion_utils.schedulers import create_scheduler


class DynamicBase(metaclass=ABCMeta):
    @abstractmethod
    def marginal_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        pass

    def reverse(self, alpha):
        pass

    @property
    def T(self):
        return 1

    @property
    def eps(self):
        return 0.001

    @staticmethod
    def prior_sampling(shape) -> Tensor:
        return torch.randn(*shape)


class DynamicSDE(DynamicBase):
    def __init__(self, config):
        """Construct a Variance Preserving SDE."""

        self.config = config
        self.N = config.dynamic.N
        self.scheduler = create_scheduler(config)

    def marginal_params(self, t: Tensor) -> Dict[str, Tensor]:
        mu, std = self.scheduler.params(t)
        return {
            "mu": mu,
            "std": std
        }

    def marginal(self, x_0: Tensor, t: Tensor, noise=None) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        params = self.marginal_params(t)
        mu, std = params["mu"], params["std"]
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = x_0 * mu + noise * std
        score = -noise / params["std"]
        return {
            "x_t": x_t,
            "noise": noise,
            "mu": mu,
            "std": std,
            "score": score,
        }

    def reverse_params(self, x_t, t, score_fn, ode_sampling=False):
        beta_t = self.scheduler.beta_t(t)
        drift_sde = (-1) / 2 * beta_t[:, None, None] * x_t
        diffuson_sde = torch.sqrt(beta_t)
        if hasattr(self.scheduler, 'delta'):
            diffuson_sde *= self.scheduler.delta
        score_output = score_fn(x_t=x_t, t=t)
        if ode_sampling:
            drift = drift_sde - (1 / 2) * beta_t[:, None, None] * score_output["score"]
            diffusion = 0
        else:
            drift = drift_sde - beta_t[:, None, None] * score_output["score"]
            diffusion = diffuson_sde
        return drift, diffusion, score_output

    def prior_sampling(self, shape) -> Tensor:
        if self.config.cluster_diffusion:
            return self.config.dynamic.delta * torch.randn(*shape)
        elif self.config.tess_diffusion:
            return self.config.dynamic.simplex_value * torch.randn(*shape)
        else:
            return torch.randn(*shape)
