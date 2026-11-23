"""Loss terms of the JointVAE. All terms are in nats per individual.

    L = recon + gamma_z |KL(q(z|x) || N(0, I)) - C_z(t)|
              + gamma_c |KL(q(c|x) || p(c))    - C_c(t)|
              + lambda_marginal KL(mean_x q(c|x) || p(c))

The first three terms are Dupont (2018). The last one is our addition (off when the weight is 0):
it asks that the categories are used as often as the prior says, so that the capacity C_c has to be
met by information about x and cannot be met by one category that every individual gets.
"""

import math

import torch
import torch.nn.functional as F


def reconstruction_loss(recon_logits, x, mask):
    """Bernoulli negative log-likelihood of the observed entries, per individual."""
    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none")
    return (bce * mask).sum(-1)


def kl_gaussian(mu, logvar):
    """KL(q(z|x) || N(0, I)) per individual and latent dimension -> [B, z_dim]."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def kl_categorical(category_logits):
    """KL(q(c|x) || uniform) per individual -> [B]."""
    log_q = F.log_softmax(category_logits, dim=-1)
    return (log_q.exp() * (log_q + math.log(category_logits.shape[-1]))).sum(-1)


def kl_categorical_marginal(category_logits):
    """KL(mean over the batch of q(c|x) || uniform) -> scalar."""
    q_bar = F.softmax(category_logits, dim=-1).mean(0).clamp_min(1e-8)
    return (q_bar * (q_bar.log() + math.log(category_logits.shape[-1]))).sum()


def capacity(step, ramp_steps, c_max):
    """Channel capacity at a training step: linear from 0 to c_max over ramp_steps."""
    if ramp_steps <= 0:
        return c_max

    return c_max * min(1.0, step / ramp_steps)


def joint_vae_objective(
    loss_recon,
    loss_kl_z,
    loss_kl_c,
    loss_kl_marginal,
    capacity_z,
    capacity_c,
    gamma_z,
    gamma_c,
    lambda_marginal,
):
    loss = loss_recon
    loss = loss + gamma_z * (loss_kl_z - capacity_z).abs()
    loss = loss + gamma_c * (loss_kl_c - capacity_c).abs()

    if lambda_marginal > 0:
        loss = loss + lambda_marginal * loss_kl_marginal

    return loss
