import abc
import torch
import numpy as np


class SDE(abc.ABC):

    def __init__(self, N):
        super().__init__()
        self.N = N

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    def reverse(self, score_fn, probability_flow=False):
        # Create the reverse-time SDE/ODE
        N = self.N
        T = self.T
        sde_fn = self.sde

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self.T = T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

        return RSDE()


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=500, T=1.0):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.T = T

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** (t/self.T)
        drift = torch.zeros_like(x)
        diffusion = sigma / self.T * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** (t/self.T)
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max


if __name__ == "__main__":
    sde = VESDE()


