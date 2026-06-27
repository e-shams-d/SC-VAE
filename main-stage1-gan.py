"""
Stage-1 SC-VAE training WITH the paper's two key components enabled:

  1. learned attention for the alpha weighting  (--attention GAT | eq | SSGC)
  2. perceptual (LPIPS) + adversarial (PatchGAN) losses

This is a self-contained driver. The repo's trainer_scvae.py is incomplete for
the current model (its active loop performs no discriminator update and reduces
the latent loss with torch.mean(list)); here we keep the SC-VAE model and data
pipeline unchanged, reuse the repo's PatchGAN discriminator + hinge/vanilla GAN
losses, and use the standard VGG-LPIPS perceptual loss (lpips package).

Generator loss : MSE(recon) + mean(alpha * A) + p_weight*LPIPS + g_weight*disc_weight*adv
Discriminator  : hinge/vanilla on (real, fake)
g_weight follows the taming adaptive-weight rule (falls back to 1.0 if unavailable).

Config keys can be overridden on the CLI, e.g.:
  python main-stage1-gan.py --attention GAT experiment.epochs=5 experiment.batch_size=4
"""
import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from scipy import linalg
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from img_datasets import create_dataset
from scvae.models import scvae
from scvae.utils.config import load_config, augment_defaults
from scvae.optimizer import create_optimizer, create_scheduler
from scvae.losses.gan.discriminator import NLayerDiscriminator, weights_init
from scvae.losses.gan.gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss

import lpips  # VGG-LPIPS perceptual loss (same method as the paper)


def build_model(cfg, num_channels, attention, device):
    """Construct the SC-VAE model exactly as in main-stage1.py (method unchanged)."""
    Hidden_size = cfg.arch.vae.hidden_size
    H_1, H_2 = cfg.arch.alpha.H_1, cfg.arch.alpha.H_2
    beta = cfg.arch.latent.get('beta', cfg.arch.latent.get('alpha'))

    Dict_init = scvae.init_dct(int(np.sqrt(Hidden_size)), 23)
    num_atoms = min(23 * 23, cfg.arch.latent.num_atoms)
    Dict_init = Dict_init[:, :num_atoms].to(device)
    c_init = torch.FloatTensor(((linalg.norm(Dict_init.cpu(), ord=2)) ** 2,)).to(device)
    w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(Hidden_size)).float().to(device)

    model = scvae.Model_VAEf16(
        cfg.arch.vae.ddconfig, num_channels, Hidden_size, H_1, H_2,
        cfg.arch.latent.num_soft_thresh, Dict_init, c_init, w_init,
        beta, device, attention,
    )
    return model.to(device)


def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    """taming-transformers adaptive GAN weight (balances adv vs recon+perceptual)."""
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    return torch.clamp(d_weight, 0.0, 1e4).detach()


def main(args, cfg):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_ds, valid_ds = create_dataset(cfg, is_eval=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.experiment.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg.experiment.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # --- SC-VAE generator (with learned attention) ---
    model = build_model(cfg, 3, args.attention, device)

    # --- PatchGAN discriminator + GAN/perceptual losses ---
    da = cfg.gan.disc.arch
    discriminator = NLayerDiscriminator(
        input_nc=da.in_channels, n_layers=da.num_layers,
        use_actnorm=da.use_actnorm, ndf=da.ndf).apply(weights_init).to(device)
    disc_loss = hinge_d_loss if cfg.gan.loss.disc_loss == 'hinge' else vanilla_d_loss
    gen_loss = vanilla_g_loss
    perceptual = lpips.LPIPS(net='vgg').to(device)
    perceptual.eval()
    for _p in perceptual.parameters():
        _p.requires_grad_(False)
    p_weight = cfg.gan.loss.perceptual_weight
    disc_weight = cfg.gan.loss.disc_weight
    disc_start = cfg.gan.loss.disc_start

    # --- optimizers / schedulers ---
    steps_per_epoch = len(train_loader)
    num_epochs = cfg.experiment.epochs

    class _Dist:  # create_scheduler only reads world_size for linear/sqrt warmup
        world_size = 1
    distenv = _Dist()

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg.optimizer.warmup, steps_per_epoch, num_epochs, distenv)
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.optimizer.init_lr, betas=tuple(cfg.optimizer.betas))
    disc_scheduler = create_scheduler(disc_optimizer, cfg.optimizer.warmup, steps_per_epoch, num_epochs, distenv)

    # --- output dirs + tensorboard ---
    date = time.strftime('%Y_%m_%d_%H_%M_%S')
    tag = 'gan_%s_ep%d_bs%d_%s' % (args.attention, num_epochs, cfg.experiment.batch_size, date)
    save_dir = os.path.join(args.dir_models, args.dataset, tag)
    log_dir = os.path.join(args.dir_logs, args.dataset, tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print('save_dir:', save_dir, flush=True)

    step = 0
    best = -1.0
    for epoch in range(num_epochs):
        model.train(); discriminator.train()
        use_disc = epoch >= disc_start
        print('epoch:' + str(epoch), flush=True)

        for images, _ in train_loader:
            images = images.to(device)

            # ---------- generator update ----------
            optimizer.zero_grad()
            x_tilde, rec, _, z = model(images)
            loss_recons = F.mse_loss(x_tilde, images)
            loss_sdl = torch.mean(rec[0] * rec[1])          # alpha * A  (correct reduction)
            loss_pcpt = perceptual(images, x_tilde).mean()  # VGG-LPIPS on [-1,1]
            nll = loss_recons + p_weight * loss_pcpt

            if use_disc:
                logits_fake, _ = discriminator(x_tilde, None)
                loss_gen = gen_loss(logits_fake)
                try:
                    g_weight = calculate_adaptive_weight(nll, loss_gen, model.get_last_layer())
                except Exception:
                    g_weight = torch.tensor(1.0, device=device)
            else:
                loss_gen = torch.zeros((), device=device)
                g_weight = torch.zeros((), device=device)

            g_total = loss_recons + loss_sdl + p_weight * loss_pcpt + g_weight * disc_weight * loss_gen
            g_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            # ---------- discriminator update ----------
            if use_disc:
                disc_optimizer.zero_grad()
                logits_fake, logits_real = discriminator(x_tilde.detach(), images.detach())
                d_loss = disc_loss(logits_real, logits_fake)
                (disc_weight * d_loss).backward()
                disc_optimizer.step(); disc_scheduler.step()
            else:
                d_loss = torch.zeros((), device=device)

            print('g_total:%.4f recon:%.4f sdl:%.6f pcpt:%.4f gen:%.4f disc:%.4f' % (
                float(g_total), float(loss_recons), float(loss_sdl),
                float(loss_pcpt), float(loss_gen), float(d_loss)), flush=True)

            writer.add_scalar('loss/train/reconstruction', loss_recons.item(), step)
            writer.add_scalar('loss/train/loss_pcpt', loss_pcpt.item(), step)
            writer.add_scalar('loss/train/loss_gen', float(loss_gen), step)
            writer.add_scalar('loss/train/loss_disc', float(d_loss), step)
            step += 1

        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            vloss, n = 0.0, 0
            for images, _ in valid_loader:
                images = images.to(device)
                x_tilde = model(images)[0]
                vloss += F.mse_loss(x_tilde, images).item(); n += 1
            vloss = vloss / max(1, n)
        print('[epoch %d] validation reconstruction loss: %.6f' % (epoch, vloss), flush=True)
        writer.add_scalar('loss/test/reconstruction', vloss, step)

        torch.save(model.state_dict(), os.path.join(save_dir, 'last.pt'))
        if best < 0 or vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

    print('done. checkpoints in', save_dir, flush=True)


def get_args():
    p = argparse.ArgumentParser(description='SC-VAE stage-1 with attention + perceptual/GAN')
    p.add_argument('--dataset', type=str, default='FFHQ')
    p.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-scvae16x16.yaml')
    p.add_argument('--attention', type=str, default='GAT', help='GAT | eq | SSGC | constant')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--dir_logs', type=str, default='./results/logs')
    p.add_argument('--dir_models', type=str, default='./results/models')
    return p.parse_known_args()


if __name__ == '__main__':
    args, extra = get_args()
    cfg = load_config(args.model_config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(extra)))
    cfg = augment_defaults(cfg)
    main(args, cfg)
