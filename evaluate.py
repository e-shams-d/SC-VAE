"""
Reconstruction evaluation for SC-VAE.

Loads a trained SC-VAE checkpoint and reports PSNR / SSIM / LPIPS on the
validation split, and writes a side-by-side (original vs reconstruction) image.

The SC-VAE model and the data pipeline are built exactly as in training/testing
(see main-stage1.py), so the method itself from the paper is unchanged -- this
script only loads the trained weights and measures reconstruction quality,
following the metric methodology of the original reconstruction.py
(PSNR/SSIM via piq, LPIPS via the lpips package, with the [-1,1] -> [0,1] remap).
"""
import os
import argparse

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from scipy import linalg

from img_datasets import create_dataset
from scvae.models import scvae
from scvae.utils.config import load_config

import piq


def build_model(model_config, num_channels, attention, device):
    """Construct the SC-VAE model exactly as in main-stage1.py (the paper's method)."""
    Hidden_size = model_config.arch.vae.hidden_size
    H_1, H_2 = model_config.arch.alpha.H_1, model_config.arch.alpha.H_2
    beta = model_config.arch.latent.get('beta', model_config.arch.latent.get('alpha'))

    Dict_init = scvae.init_dct(int(np.sqrt(Hidden_size)), 23)
    num_atoms = min(23 * 23, model_config.arch.latent.num_atoms)
    Dict_init = Dict_init[:, :num_atoms].to(device)
    c_init = torch.FloatTensor(((linalg.norm(Dict_init.cpu(), ord=2)) ** 2,)).to(device)
    w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(Hidden_size)).float().to(device)

    model = scvae.Model_VAEf16(
        model_config.arch.vae.ddconfig, num_channels, Hidden_size, H_1, H_2,
        model_config.arch.latent.num_soft_thresh, Dict_init, c_init, w_init,
        beta, device, attention,
    )
    return model.to(device)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_config = load_config(args.model_config)

    # validation split, using the exact same data pipeline as training/testing
    _, valid_dataset = create_dataset(model_config, is_eval=True)
    loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # build the SC-VAE model and load the trained weights
    model = build_model(model_config, num_channels=3, attention=args.attention, device=device)
    model.load_state_dict(torch.load(args.load_path, map_location=device))
    model.eval()
    print('loaded checkpoint:', args.load_path, flush=True)

    # LPIPS is optional (matches the original reconstruction.py, which used the lpips package)
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        print('LPIPS unavailable (pip install lpips), skipping it:', e)

    os.makedirs(args.out_dir, exist_ok=True)
    psnrs, ssims, lp_vals = [], [], []
    n_images = 0
    saved = False

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            x_tilde = model(images)[0]          # forward -> (x_tilde, rec, Dict, z)

            # LPIPS on the native [-1, 1] range
            if lpips_fn is not None:
                lp_vals.append(lpips_fn(images, x_tilde).reshape(-1))

            # PSNR / SSIM on [0, 1] (same remap as the original reconstruction.py)
            img01 = torch.clamp(images * 0.5 + 0.5, 0, 1)
            rec01 = torch.clamp(x_tilde * 0.5 + 0.5, 0, 1)
            psnrs.append(piq.psnr(img01, rec01, data_range=1., reduction='none'))
            ssims.append(piq.ssim(img01, rec01, data_range=1., reduction='none'))
            n_images += images.size(0)

            # save one side-by-side comparison (top: originals, bottom: reconstructions)
            if not saved:
                k = min(8, img01.size(0))
                grid = make_grid(torch.cat([img01[:k], rec01[:k]], dim=0).cpu(), nrow=k)
                save_image(grid, os.path.join(args.out_dir, 'reconstruction_compare.png'))
                saved = True

    print('evaluated on %d images' % n_images)
    print('PSNR : %.4f' % torch.cat(psnrs).mean().item())
    print('SSIM : %.4f' % torch.cat(ssims).mean().item())
    if lpips_fn is not None:
        print('LPIPS: %.4f' % torch.cat(lp_vals).mean().item())
    print('saved comparison image:', os.path.join(args.out_dir, 'reconstruction_compare.png'), flush=True)


def get_args():
    p = argparse.ArgumentParser(description='SC-VAE reconstruction evaluation')
    p.add_argument('--dataset', type=str, default='FFHQ')
    p.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-scvae16x16.yaml')
    p.add_argument('--load-path', type=str, required=True, help='trained checkpoint (best.pt / last.pt)')
    p.add_argument('--attention', type=str, default='constant', help='must match the training run')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--out-dir', type=str, default='./results/eval')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
