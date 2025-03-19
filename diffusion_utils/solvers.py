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
        !!!!!!!!NON-MARKOV FOR NOW!!!!!!!!!
        """

        score_output = self.score_fn(x_t=x_t, t=t, **kwargs)

        x = self.dynamic.marginal(score_output['x_0'], next_t)['x_t']

        # DDPM
        # params = self.dynamic.marginal_params(t)
        # alpha_bar_t = params['mu']
        # params = self.dynamic.marginal_params(next_t)
        # alpha_bar_t_next, std_next = params['mu'], params['std']
        # alpha_t = alpha_bar_t / alpha_bar_t_next

        # noise_var = std_next ** 2 * (1 - alpha_t) / (1 - alpha_bar_t)

        # mu_x_t_coef = alpha_t ** 0.5 * (1 - alpha_bar_t_next) / (1 - alpha_bar_t)
        # mu_x_0_coef = alpha_bar_t_next ** 0.5 * (1 - alpha_t) / (1 - alpha_bar_t)
        # mu = mu_x_t_coef * x_t + mu_x_0_coef * score_output['x_0']

        # noise = torch.randn_like(x_t)
        # x = mu + noise_var ** 0.5 * noise

        # SDE
        # dt = (next_t - t).view(-1, 1, 1)
        # drift, diffusion, score_output = self.dynamic.reverse_params(
        #     x_t, t, partial(self.score_fn, **kwargs), self.ode_sampling
        # )
        # x_mean = x_t + drift * dt
        # noise = torch.randn_like(x_t)
        # x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
        return {
            "x": x,
            "x_mean": x,
            "x_0": score_output["x_0"],
            "latent_pred": score_output["latent_pred"]
        }
