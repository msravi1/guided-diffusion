"""
VP-SDE Score Model Implementation
EE/CS 148B HW4 - Part 5

Variance Preserving SDE:
    dx = -1/2 * beta(t) * x dt + sqrt(beta(t)) dBt

where beta(t) = beta_min + (beta_max - beta_min) * t  (linear schedule)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class VPSDE:
    """
    Variance Preserving SDE.

    Forward process:
        dx = -1/2 * beta(t) * x dt + sqrt(beta(t)) dBt

    Marginal distribution:
        x(t) | x(0) ~ N( c(t)*x(0),  sigma(t)^2 * I )
    where
        c(t)      = exp( -1/2 * int_0^t beta(s) ds )
        sigma(t)^2 = 1 - c(t)^2            (variance preserving property)
    """

    def __init__(
        self,
        beta_min: float = 0.01,
        beta_max: float = 5.0,
        T: float = 1.0,
        device: str = "cpu",
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.device = device

    # ------------------------------------------------------------------
    # Beta schedule and derived quantities
    # ------------------------------------------------------------------

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule: beta(t) = beta_min + (beta_max - beta_min)*t"""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Integral of beta from 0 to t:
            B(t) = beta_min*t + (beta_max - beta_min)*t^2/2
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def c(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mean scaling coefficient:
            c(t) = exp( -1/2 * B(t) )
        This matches Eq. (33) in Song et al. 2021.
        """
        return torch.exp(-0.5 * self.int_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Standard deviation of the marginal:
            sigma(t) = sqrt( 1 - c(t)^2 )
        Variance preserving: Var[x(t)] = c(t)^2 + sigma(t)^2 = 1 if x(0) ~ N(0,I).
        """
        return torch.sqrt(torch.clamp(1.0 - self.c(t) ** 2, min=1e-10))

    # ------------------------------------------------------------------
    # Drift and diffusion coefficients of the forward SDE
    # ------------------------------------------------------------------

    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient: f(x, t) = -1/2 * beta(t) * x
        Broadcast beta(t) over spatial dims.
        """
        # t shape: (B,), x shape: (B, C, H, W)
        bt = self.beta(t).view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * bt * x

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient: g(t) = sqrt(beta(t))
        """
        return torch.sqrt(self.beta(t))

    # ------------------------------------------------------------------
    # Forward process: sample x(t) given x(0)
    # ------------------------------------------------------------------

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, std) of p(x(t) | x(0)).
            mean = c(t) * x0
            std  = sigma(t)   [scalar broadcast]
        """
        ct = self.c(t).view(-1, *([1] * (x0.dim() - 1)))
        st = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        return ct * x0, st

    def sample_forward(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x(t) ~ p(x(t)|x(0)) via reparameterisation:
            x(t) = c(t)*x0 + sigma(t)*eps,  eps ~ N(0, I)
        Returns (x_t, eps).
        """
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        x_t = mean + std * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Score function from network output
    # ------------------------------------------------------------------

    def score_from_eps(
        self, eps_theta: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert predicted noise eps_theta to score:
            score = -eps_theta / sigma(t)
        (from Tweedie / DSM equivalence)
        """
        st = self.sigma(t).view(-1, *([1] * (eps_theta.dim() - 1)))
        return -eps_theta / st

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def loss(
        self,
        score_model: nn.Module,
        x0: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        DSM loss: E[ || s_theta(x_t, t) + eps/sigma(t) ||^2 ]
        Equivalently (and more numerically stable): MSE on predicted noise.
            L = E[ || eps_theta(x_t, t) - eps ||^2 ]
        """
        B = x0.shape[0]
        # Sample t ~ Uniform(eps, T)
        t = torch.rand(B, device=x0.device) * (self.T - eps) + eps
        x_t, eps_sample = self.sample_forward(x0, t)
        # Model predicts the noise
        eps_pred = score_model(x_t, t)
        return torch.mean((eps_pred - eps_sample) ** 2)

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_maruyama_sampler(
        self,
        score_model: nn.Module,
        shape: Tuple,
        num_steps: int = 1000,
        eps: float = 1e-3,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Euler-Maruyama discretisation of the reverse VP-SDE.

        Reverse SDE (Anderson 1982 / Song 2021 Eq. 6):
            dx = [ f(x,t) - g(t)^2 * score(x,t) ] dt + g(t) dB_bar_t

        EM update (going from t -> t - dt, i.e. backwards):
            x_{t-dt} = x_t - [ f(x_t, t) - g(t)^2 * score ] * dt
                        + g(t) * sqrt(dt) * z,   z ~ N(0,I)
        """
        score_model.eval()

        # Initial sample: x(T) ~ N(0, sigma(T)^2 * I)
        t_init = torch.ones(1, device=device) * self.T
        sigma_T = self.sigma(t_init).item()
        x = torch.randn(shape, device=device) * sigma_T

        # Time grid: T -> eps (backwards)
        time_steps = torch.linspace(self.T, eps, num_steps + 1, device=device)
        dt = (self.T - eps) / num_steps  # positive step size

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_batch = t_cur.expand(shape[0])

            # Predict noise, convert to score
            eps_pred = score_model(x, t_batch)
            score = self.score_from_eps(eps_pred, t_batch)

            # Drift and diffusion
            drift = self.f(x, t_batch) - self.g(t_batch).view(-1, 1, 1, 1) ** 2 * score
            diffusion = self.g(t_batch).view(-1, 1, 1, 1)

            # EM step (reverse in time, so subtract drift*dt)
            z = torch.randn_like(x)
            x = x - drift * dt + diffusion * np.sqrt(dt) * z

        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def predictor_corrector_sampler(
        self,
        score_model: nn.Module,
        shape: Tuple,
        num_steps: int = 1000,
        num_corrector_steps: int = 1,
        snr: float = 0.16,
        eps: float = 1e-3,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Predictor-Corrector sampler (Algorithm 5 in Song et al. 2021).

        Predictor: Euler-Maruyama reverse step.
        Corrector: Annealed Langevin dynamics (score-based MCMC).

        Args:
            num_corrector_steps: number of Langevin steps per predictor step
            snr: signal-to-noise ratio for Langevin step size
        """
        score_model.eval()

        # Initial sample
        t_init = torch.ones(1, device=device) * self.T
        sigma_T = self.sigma(t_init).item()
        x = torch.randn(shape, device=device) * sigma_T

        time_steps = torch.linspace(self.T, eps, num_steps + 1, device=device)
        dt = (self.T - eps) / num_steps

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_batch = t_cur.expand(shape[0])

            # ---- Corrector: Langevin dynamics at current t ----
            for _ in range(num_corrector_steps):
                eps_pred = score_model(x, t_batch)
                score = self.score_from_eps(eps_pred, t_batch)

                # Adaptive step size (Algorithm 5 of Song 2021)
                grad_norm = torch.norm(
                    score.reshape(shape[0], -1), dim=-1
                ).mean()
                noise_norm = np.sqrt(np.prod(shape[1:]))
                alpha = 2 * (snr * noise_norm / (grad_norm + 1e-8)) ** 2

                # Langevin update
                z = torch.randn_like(x)
                x = x + alpha * score + torch.sqrt(2 * alpha) * z

            # ---- Predictor: EM reverse step ----
            eps_pred = score_model(x, t_batch)
            score = self.score_from_eps(eps_pred, t_batch)

            drift = self.f(x, t_batch) - self.g(t_batch).view(-1, 1, 1, 1) ** 2 * score
            diffusion = self.g(t_batch).view(-1, 1, 1, 1)

            z = torch.randn_like(x)
            x = x - drift * dt + diffusion * np.sqrt(dt) * z

        return x.clamp(-1.0, 1.0)
