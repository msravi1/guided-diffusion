"""
Train VP-SDE Score Model on FashionMNIST
EE/CS 148B HW4 - Part 5
"""

import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import your VP SDE and UNet (adjust path as needed)
from vp import VPSDE
# from unet import UNet  # provided by starter code


# -----------------------------------------------------------------------
# Tiny UNet stub (replace with the provided time-conditioned UNet)
# -----------------------------------------------------------------------
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeCondUNet(nn.Module):
    """
    Minimal time-conditioned UNet for 28x28 grayscale images.
    Replace with the starter-code UNet for full performance.
    """
    def __init__(self, channels=32, t_emb_dim=128):
        super().__init__()
        self.t_emb = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Linear(t_emb_dim, channels)

        self.enc1 = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(channels, channels*2, 3, padding=1, stride=2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(channels*2, channels*4, 3, padding=1, stride=2), nn.SiLU())
        self.mid  = nn.Sequential(nn.Conv2d(channels*4, channels*4, 3, padding=1), nn.SiLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(channels*4, channels*2, 4, stride=2, padding=1), nn.SiLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels*4, channels, 4, stride=2, padding=1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv2d(channels*2, channels, 3, padding=1), nn.SiLU())
        self.out  = nn.Conv2d(channels, 1, 1)

    def forward(self, x, t):
        te = self.t_proj(self.t_emb(t))  # (B, channels)
        # add time embedding to first feature map
        e1 = self.enc1(x) + te.view(-1, te.shape[-1], 1, 1)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m  = self.mid(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out(d1)


# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------

def get_fashion_mnist(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # map to [-1, 1]
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def visualize_dataset(loader, save_path="dataset_samples.png"):
    imgs, labels = next(iter(loader))
    imgs = imgs[:64]
    grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.title("FashionMNIST Dataset Samples (64 images)\n"
              "Classes: T-shirt, Trouser, Pullover, Dress, Coat, "
              "Sandal, Shirt, Sneaker, Bag, Ankle boot\n"
              "Image dimensions: 1 × 28 × 28 (grayscale)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dataset visualization saved to {save_path}")


# -----------------------------------------------------------------------
# Coefficient plot (Problem 1.8)
# -----------------------------------------------------------------------

def plot_ddpm_coefficient(save_path="ddpm_coefficient.png"):
    """
    Plot the DDPM loss coefficient:
        beta_t^2 / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))
    where sigma_t^2 = beta_t and the linear beta schedule is used.
    """
    T = 1000
    beta_1, beta_T = 1e-4, 0.02
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    sigma2 = betas  # sigma_t^2 = beta_t (as given)

    coeff = betas ** 2 / (2 * sigma2 * alphas * (1.0 - alpha_bar))

    t = np.arange(1, T + 1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(t, coeff, color="#2c7bb6", linewidth=2)
    plt.xlabel("Timestep $t$", fontsize=14)
    plt.ylabel(r"$\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$  (log scale)", fontsize=14)
    plt.title("DDPM Loss Coefficient vs. Timestep (Problem 1.8)", fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Coefficient plot saved to {save_path}")


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_vp(
    beta_min: float = 0.01,
    beta_max: float = 5.0,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cuda",
    save_dir: str = "./checkpoints_vp",
):
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_loader, _ = get_fashion_mnist(batch_size)

    # Model and SDE
    model = TimeCondUNet(channels=64).to(device)
    sde   = VPSDE(beta_min=beta_min, beta_max=beta_max, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            loss = sde.loss(model, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss:.6f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vp.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "final_vp.pt"))

    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(train_losses) + 1), train_losses,
                 color="#d7191c", linewidth=2, label="Train Loss")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("MSE Loss (log scale)", fontsize=13)
    plt.title(f"VP-SDE Training Loss (β_min={beta_min}, β_max={beta_max})", fontsize=13)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vp_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return model, sde, train_losses


# -----------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------

def save_sample_grid(samples: torch.Tensor, path: str, title: str = ""):
    grid = torchvision.utils.make_grid(
        samples[:64].cpu(), nrow=8, normalize=True, value_range=(-1, 1)
    )
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_min", type=float, default=0.01)
    parser.add_argument("--beta_max", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_steps_em", type=int, default=500)
    parser.add_argument("--num_steps_pc", type=int, default=500)
    parser.add_argument("--corrector_steps", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--save_dir", type=str, default="./outputs_vp")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Dataset visualization (Problem 5.C.i)
    train_loader, _ = get_fashion_mnist(args.batch_size)
    visualize_dataset(train_loader, os.path.join(args.save_dir, "dataset_samples.png"))

    # 2) Coefficient plot (Problem 1.8)
    plot_ddpm_coefficient(os.path.join(args.save_dir, "ddpm_coefficient.png"))

    # 3) Train
    model, sde, losses = train_vp(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir,
    )

    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_vp.pt")))
    model.eval()

    shape = (64, 1, 28, 28)

    # 4) EM samples (Problem 5.C.iii)
    print("Generating EM samples...")
    em_samples = sde.euler_maruyama_sampler(
        model, shape, num_steps=args.num_steps_em, device=args.device
    )
    save_sample_grid(em_samples,
                     os.path.join(args.save_dir, "em_samples.png"),
                     f"VP-SDE Euler-Maruyama Samples ({args.num_steps_em} steps)")

    # 5) PC samples (Problem 5.C.iv)
    for nc in args.corrector_steps:
        print(f"Generating PC samples (corrector_steps={nc})...")
        pc_samples = sde.predictor_corrector_sampler(
            model, shape,
            num_steps=args.num_steps_pc,
            num_corrector_steps=nc,
            device=args.device,
        )
        save_sample_grid(pc_samples,
                         os.path.join(args.save_dir, f"pc_samples_nc{nc}.png"),
                         f"VP-SDE PC Samples ({args.num_steps_pc} steps, {nc} corrector steps)")

    print("Done! All outputs saved to", args.save_dir)
