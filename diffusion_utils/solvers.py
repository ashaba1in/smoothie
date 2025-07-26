import torch


def create_solver():
    return Solver


class Solver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, solver=None, **kwargs):
        score_output = self.score_fn(x_t=x_t, t=t, **kwargs)

        if solver == 'ddpm':
            params = self.dynamic.marginal_params(t)
            alpha_bar_t = params['mu']**2
            params = self.dynamic.marginal_params(next_t)
            alpha_bar_t_next, std_next = params['mu']**2, params['std']
            alpha_t = alpha_bar_t / alpha_bar_t_next

            noise_var = std_next ** 2 * (1 - alpha_t) / (1 - alpha_bar_t)

            mu_x_t_coef = alpha_t ** 0.5 * (1 - alpha_bar_t_next) / (1 - alpha_bar_t)
            mu_x_0_coef = alpha_bar_t_next ** 0.5 * (1 - alpha_t) / (1 - alpha_bar_t)
            mu = mu_x_t_coef * x_t + mu_x_0_coef * score_output['x_0']

            noise = torch.randn_like(x_t)
            x = mu + noise_var ** 0.5 * noise
        elif solver == 'ddim':
            params = self.dynamic.marginal_params(t)
            pred_eps = (x_t - params['mu'] * score_output['x_0']) / params['std']
            x = self.dynamic.marginal(score_output['x_0'], next_t, noise=pred_eps)['x_t']
        elif solver == 'sde':
            beta_t = self.dynamic.scheduler.beta_t(t)
            drift_sde = (-1) / 2 * beta_t[:, None, None] * x_t
            diffusion = torch.sqrt(beta_t)
            drift = drift_sde - beta_t[:, None, None] * score_output['score']
            
            dt = (next_t - t)[:, None, None]
            x_mean = x_t + drift * dt

            noise = torch.randn_like(x_t)
            x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
        else:
            x = self.dynamic.marginal(score_output['x_0'], next_t)['x_t']

        return {
            "x": x,
            "x_0": score_output["x_0"],
            "latent_pred": score_output["latent_pred"]
        }
    
# class DynamicSDE(DynamicBase):
#     def __init__(self, config):
#         """Construct a Variance Preserving SDE.

#         Args:
#           beta_min: value of beta(0)
#           beta_max: value of beta(1)
#           N: number of discretization steps
#         """

#         self.N = config.dynamic.N
#         self.scheduler = create_scheduler(config)

#     def marginal_params(self, t: Tensor) -> Dict[str, Tensor]:
#         mu, std = self.scheduler.params(t)
#         return {
#             "mu": mu,
#             "std": std
#         }

#     def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
#         """
#         Calculate marginal q(x_t|x_0)'s mean and std
#         """
#         params = self.marginal_params(t)
#         mu, std = params["mu"], params["std"]
#         noise = torch.randn_like(x_0)
#         x_t = x_0 * mu + noise * std
#         score = -noise / params["std"]
#         return {
#             "x_t": x_t,
#             "noise": noise,
#             "mu": mu,
#             "std": std,
#             "score": score,
#         }

#     def reverse_params(self, x_t, t, score_fn, ode_sampling=False):
#         beta_t = self.scheduler.beta_t(t)
#         drift_sde = (-1) / 2 * beta_t[:, None, None] * x_t
#         diffuson_sde = torch.sqrt(beta_t)
#         score_output = score_fn(x_t=x_t, t=t)
        
#         drift = drift_sde - beta_t[:, None, None] * score_output['score']
#         diffusion = diffuson_sde
#         return drift, diffusion, score_output
