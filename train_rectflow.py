"""
Train Rectified Flow on FashionMNIST
EE/CS 148B HW4 - Part 6
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from rectflow import RectifiedFlow

# Use the same UNet stub as in train_vp.py
# In practice, import from the starter code:
# from unet import UNet
import math
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
        te = self.t_proj(self.t_emb(t))
        e1 = self.enc1(x) + te.view(-1, te.shape[-1], 1, 1)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m  = self.mid(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out(d1)


# -----------------------------------------------------------------------
# Data helpers (same as train_vp.py)
# -----------------------------------------------------------------------

def get_fashion_mnist(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    return train_loader


def save_sample_grid(samples, path, title=""):
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


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_rectflow(
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cuda",
    save_dir: str = "./outputs_rf",
    reflow: bool = False,
    reflow_pairs_path: str = None,
    reflow_epochs: int = 20,
):
    os.makedirs(save_dir, exist_ok=True)

    rf = RectifiedFlow(device=device)

    if reflow and reflow_pairs_path is not None:
        # Load pre-generated reflow pairs
        pairs = torch.load(reflow_pairs_path)
        x0_pairs, x1_pairs = pairs["x0"], pairs["x1"]
        dataset = TensorDataset(x0_pairs, x1_pairs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
        total_epochs = reflow_epochs
        label = "reflow"
    else:
        loader = get_fashion_mnist(batch_size)
        total_epochs = epochs
        label = "rf"

    model = TimeCondUNet(channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    losses = []
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            if reflow:
                x0_b, x1_b = batch[0].to(device), batch[1].to(device)
                t = torch.rand(x0_b.shape[0], device=device)
                loss = rf.reflow_loss(model, x0_b, x1_b, t)
            else:
                x1_b = batch[0].to(device)
                loss = rf.loss(model, x1_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x1_b.size(0)

        n = len(loader.dataset)
        epoch_loss /= n
        losses.append(epoch_loss)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{label}.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, f"final_{label}.pt"))
    return model, rf, losses


# -----------------------------------------------------------------------
# Combined loss curve plot
# -----------------------------------------------------------------------

def plot_combined_losses(vp_losses, rf_losses, save_path):
    plt.figure(figsize=(9, 5))
    plt.semilogy(vp_losses, label="VP-SDE (DDPM)", color="#d7191c", linewidth=2)
    plt.semilogy(rf_losses, label="Rectified Flow", color="#2c7bb6", linewidth=2)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("MSE Loss (log scale)", fontsize=13)
    plt.title("Training Loss: VP-SDE vs. Rectified Flow", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined loss plot saved to {save_path}")


# -----------------------------------------------------------------------
# Side-by-side qualitative grid (Problem 6.D)
# -----------------------------------------------------------------------

def qualitative_grid(
    models_dict: dict,  # {label: (model_fn, num_steps)}
    fixed_seeds: torch.Tensor,
    save_path: str,
    device: str = "cuda",
):
    """
    Generate a (n_methods x 8) grid from fixed initial noise vectors.
    """
    n_seeds = fixed_seeds.shape[0]
    rows = []

    for label, (gen_fn, _) in models_dict.items():
        row = gen_fn(fixed_seeds.to(device))  # (8, C, H, W)
        rows.append(row.cpu())

    # Stack into grid
    all_imgs = torch.cat(rows, dim=0)  # (n_methods * 8, C, H, W)
    grid = torchvision.utils.make_grid(
        all_imgs, nrow=n_seeds, normalize=True, value_range=(-1, 1)
    )
    plt.figure(figsize=(16, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    method_labels = list(models_dict.keys())
    plt.title("  |  ".join(method_labels), fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Qualitative grid saved to {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./outputs_rf")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--do_reflow", action="store_true")
    parser.add_argument("--n_reflow_pairs", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Train rectified flow
    print("=== Training Rectified Flow ===")
    model_rf, rf, rf_losses = train_rectflow(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir,
    )

    # Load best checkpoint
    model_rf.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_rf.pt"), map_location=args.device)
    )
    model_rf.eval()

    # 2) Euler samples at various step counts (Problem 6.B)
    shape = (64, 1, 28, 28)
    for n_steps in [1, 5, 10, 50, 100]:
        print(f"Sampling RF with {n_steps} Euler steps...")
        samples = rf.euler_sampler(model_rf, shape, num_steps=n_steps, device=args.device)
        save_sample_grid(
            samples,
            os.path.join(args.save_dir, f"rf_euler_{n_steps}steps.png"),
            f"Rectified Flow — {n_steps} Euler steps"
        )

    # 3) Reflow (Problem 6.C)
    if args.do_reflow:
        print("=== Generating reflow pairs ===")
        pairs_path = os.path.join(args.save_dir, "reflow_pairs.pt")
        x0_pairs, x1_pairs = rf.generate_reflow_pairs(
            model_rf,
            n_pairs=args.n_reflow_pairs,
            shape_single=(1, 28, 28),
            num_ode_steps=100,
            device=args.device,
        )
        torch.save({"x0": x0_pairs, "x1": x1_pairs}, pairs_path)
        print(f"Saved {args.n_reflow_pairs} reflow pairs to {pairs_path}")

        print("=== Reflow training ===")
        model_reflow, rf2, reflow_losses = train_rectflow(
            device=args.device,
            save_dir=args.save_dir,
            reflow=True,
            reflow_pairs_path=pairs_path,
            reflow_epochs=20,
        )
        model_reflow.load_state_dict(
            torch.load(os.path.join(args.save_dir, "best_reflow.pt"), map_location=args.device)
        )
        model_reflow.eval()

        # 1-step reflow sample
        samples_reflow1 = rf2.euler_sampler(model_reflow, shape, num_steps=1, device=args.device)
        save_sample_grid(
            samples_reflow1,
            os.path.join(args.save_dir, "reflow_1step.png"),
            "Reflow — 1 Euler step"
        )

    print("Rectified Flow training and sampling complete!")
    print("Results saved to:", args.save_dir)
