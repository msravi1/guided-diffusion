"""
Rectified Flow Implementation
EE/CS 148B HW4 - Part 6

Learns a straight-line ODE transport from noise to data:
    X_t = (1-t)*X_0 + t*X_1,  t in [0,1]
    X_0 ~ N(0,I),  X_1 ~ p_data

Training loss:
    L = E[ || (X_1 - X_0) - v_theta(X_t, t) ||^2 ]

Sampling ODE:
    dX_t/dt = v_theta(X_t, t),  X_0 ~ N(0,I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class RectifiedFlow:
    """
    Rectified Flow generative model.

    The forward process linearly interpolates between noise and data:
        X_t = (1-t)*noise + t*data

    The model v_theta learns to predict the velocity X_1 - X_0.
    At inference, we integrate the ODE forward from t=0 to t=1.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def forward_process(
        self, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample x_t along the linear interpolation path.

        Args:
            x1: clean data samples, shape (B, C, H, W)
            t:  time values in [0,1], shape (B,)

        Returns:
            x_t:  interpolated sample,    shape (B, C, H, W)
            x0:   noise sample,           shape (B, C, H, W)
            vel:  velocity target X1-X0,  shape (B, C, H, W)
        """
        x0 = torch.randn_like(x1)  # noise

        # Broadcast t over spatial dimensions
        t_expand = t.view(-1, *([1] * (x1.dim() - 1)))

        x_t = (1.0 - t_expand) * x0 + t_expand * x1
        vel = x1 - x0  # regression target

        return x_t, x0, vel

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def loss(self, v_model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
        """
        Rectified Flow MSE loss.

        L(theta) = E_{t~U[0,1], x0~N(0,I), x1~p_data}[
            || (x1 - x0) - v_theta(x_t, t) ||^2
        ]
        """
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)  # t ~ Uniform(0, 1)

        x_t, x0, vel = self.forward_process(x1, t)
        vel_pred = v_model(x_t, t)

        return F.mse_loss(vel_pred, vel)

    # ------------------------------------------------------------------
    # Euler ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sampler(
        self,
        v_model: nn.Module,
        shape: Tuple,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Euler method for the rectified flow ODE:
            x_{t+dt} = x_t + v_theta(x_t, t) * dt

        Integrates from t=0 (noise) to t=1 (data).

        Args:
            v_model:   trained velocity network
            shape:     output shape (B, C, H, W)
            num_steps: number of Euler steps
            device:    torch device

        Returns:
            Generated samples at t=1, shape (B, C, H, W)
        """
        v_model.eval()

        # Start from pure noise at t=0
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t_batch = torch.full((shape[0],), t_val, device=device)

            # Velocity prediction
            v = v_model(x, t_batch)

            # Euler step
            x = x + v * dt

        return x.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Reflow: generate new (noise, data) pairs for retraining
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        v_model: nn.Module,
        n_pairs: int,
        shape_single: Tuple,  # (C, H, W)
        num_ode_steps: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate coupled (X0_hat, X1_hat) pairs for the reflow procedure.

        Steps:
          1. Sample fresh noise X0_hat ~ N(0, I)
          2. Run the ODE forward to get X1_hat = Phi^1(X0_hat)
          3. The new training pairs are (X0_hat, X1_hat)

        These re-paired samples produce straighter trajectories because
        X1_hat is the deterministic image produced from X0_hat, so the
        coupling is "monge-optimal"-like and minimises crossing.
        """
        v_model.eval()

        all_x0 = []
        all_x1 = []

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            B = end - start
            shape = (B,) + shape_single

            x0 = torch.randn(shape, device=device)
            all_x0.append(x0.cpu())

            # Integrate ODE
            x = x0.clone()
            dt = 1.0 / num_ode_steps
            for i in range(num_ode_steps):
                t_val = i * dt
                t_batch = torch.full((B,), t_val, device=device)
                v = v_model(x, t_batch)
                x = x + v * dt

            all_x1.append(x.clamp(-1.0, 1.0).cpu())

        return torch.cat(all_x0, dim=0), torch.cat(all_x1, dim=0)

    def reflow_loss(
        self,
        v_model: nn.Module,
        x0_pairs: torch.Tensor,
        x1_pairs: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reflow training loss on pre-generated (X0_hat, X1_hat) pairs.

        Uses same RF objective but with the re-paired dataset, which
        has straighter coupling than the original independent pairing.
        """
        B = x0_pairs.shape[0]
        if t is None:
            t = torch.rand(B, device=x0_pairs.device)

        t_expand = t.view(-1, *([1] * (x0_pairs.dim() - 1)))
        x_t = (1.0 - t_expand) * x0_pairs + t_expand * x1_pairs
        vel_target = x1_pairs - x0_pairs

        vel_pred = v_model(x_t, t)
        return F.mse_loss(vel_pred, vel_target)


# Make Optional available at module level
from typing import Optional
