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

    @property
    def T(self):
        return 1.0

    @property
    def eps(self):
        return 0.0

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

    def prior_sampling(self, shape) -> Tensor:
        if self.config.dynamic.scheduler == 'arctan':
            return self.config.dynamic.delta * torch.randn(*shape)
        elif self.config.dynamic.scheduler == 'tess':
            return self.config.dynamic.simplex_value * torch.randn(*shape)
        else:
            return torch.randn(*shape)
