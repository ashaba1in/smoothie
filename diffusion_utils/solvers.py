import torch
from functools import partial


def create_solver(config):
    if config.dynamic.solver == "euler":
        return EulerDiffEqSolver


class EulerDiffEqSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = (next_t - t).view(-1, 1, 1)
        drift, diffusion, score_output = self.dynamic.reverse_params(
            x_t, t, partial(self.score_fn, **kwargs), self.ode_sampling
        )
        x_mean = x_t + drift * dt
        noise = torch.randn_like(x_t)
        x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
        return {
            "x": x,
            "x_mean": x_mean,
            "x_0": score_output["x_0"],
        }
