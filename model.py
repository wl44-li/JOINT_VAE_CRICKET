"""JointVAE (Dupont, 2018) with an MLP backbone for cricket SNP windows.

A Gaussian latent z and one K-way categorical latent c (Gumbel-softmax). The backbone has the
widths of the CAD-VAE and cVAE models for the same data (512-128, 16 Gaussian latents).

Input: every SNP is two binary channels (ALT presence, REF presence), as in the stored zarr files.
A missing call is [0, 0] there; here it is masked: replaced by the training mean in the encoder
input and left out of the reconstruction loss.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MISSING = 3


@dataclass
class ModelConfig:
    z_dim: int = 16
    n_categories: int = 3

    hidden_dims: tuple = (512, 128)
    dropout: float = 0.1
    temperature: float = 0.67


def encode_genotype(dosage):
    """uint8 dosage [B, n] (3 = missing) -> channels [B, 2n] and the mask of observed entries."""
    x = torch.stack(((dosage == 1) | (dosage == 2), dosage < 2), dim=-1)
    x = x.to(torch.float32).flatten(1)
    mask = (dosage != MISSING).repeat_interleave(2, dim=1).to(torch.float32)
    return x, mask


def mlp(input_dim, hidden_dims, dropout):
    """Build MLP hidden layers without a final output layer."""
    layers = []

    for width in hidden_dims:
        layers += [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        input_dim = width

    return nn.Sequential(*layers)


class JointVAE(nn.Module):
    """q(z, c | x) = q(z | x) q(c | x); p(x | z, c) Bernoulli; p(z) = N(0, I); p(c) uniform."""

    def __init__(self, input_dim, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()

        self.cfg = cfg
        self.input_dim = input_dim
        self.z_dim = cfg.z_dim
        self.n_categories = cfg.n_categories

        self.encoder = mlp(input_dim, cfg.hidden_dims, cfg.dropout)

        hidden = cfg.hidden_dims[-1]
        self.fc_mu = nn.Linear(hidden, cfg.z_dim)
        self.fc_logvar = nn.Linear(hidden, cfg.z_dim)
        self.fc_category = nn.Linear(hidden, cfg.n_categories)

        self.decoder = nn.Sequential(
            mlp(cfg.z_dim + cfg.n_categories, tuple(reversed(cfg.hidden_dims)), cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], input_dim),
        )

        # Training mean of every input channel (the allele-frequency information).
        self.register_buffer("input_mean", torch.zeros(input_dim))

    @torch.no_grad()
    def set_data_stats(self, feature_mean):
        """Centre the encoder input and start the decoder at the allele frequencies."""
        self.input_mean.copy_(feature_mean)

        p = feature_mean.clamp(1e-3, 1 - 1e-3)
        self.decoder[-1].bias.copy_(torch.log(p) - torch.log1p(-p))

    def encode(self, x, mask):
        """Return (mu, logvar, category logits); missing entries enter as the training mean."""
        hidden = self.encoder(torch.where(mask > 0, x, self.input_mean) - self.input_mean)
        return self.fc_mu(hidden), self.fc_logvar(hidden), self.fc_category(hidden)

    def sample(self, mu, logvar, category_logits, sample=True):
        """Training: reparameterised z and relaxed c. Evaluation: z = mu, c = one-hot(arg-max)."""
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            c = F.gumbel_softmax(category_logits, tau=self.cfg.temperature, hard=False)
        else:
            z = mu
            c = F.one_hot(category_logits.argmax(-1), self.n_categories).to(mu.dtype)

        return torch.cat([z, c], dim=-1)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x, mask, sample=True):
        mu, logvar, category_logits = self.encode(x, mask)
        recon_logits = self.decode(self.sample(mu, logvar, category_logits, sample))
        return recon_logits, mu, logvar, category_logits
