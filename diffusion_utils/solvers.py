def create_solver():
    return Solver


class Solver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        score_output = self.score_fn(x_t=x_t, t=t, **kwargs)
        x = self.dynamic.marginal(score_output['x_0'], next_t)['x_t']

        return {
            "x": x,
            "x_0": score_output["x_0"],
            "latent_pred": score_output["latent_pred"]
        }
