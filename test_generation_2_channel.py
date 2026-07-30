#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import torch
import numpy as np
import os
import torchvision
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pathlib import Path

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from torchvision import transforms as Transform
from param_normalization import normalize as pn_normalize, load_stats as pn_load_stats, PARAM_NAMES

#%% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):

        _, _, self.betas = get_sigma_schedule(args, device=device)

        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped


    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt, param_values=None):
    """Sample from the model with conditional parameters"""
    x = x_init
    with torch.no_grad():
        for i in tqdm(reversed(range(n_time)), desc="Generating", leave=False):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)

            # Pass param_values to generator
            x_0 = generator(x, t_time, latent_z, param=param_values)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

def load_training_parameters(properties_txt="properties_grs_cut.txt"):
    """Load all training parameters (one row per sample: z, logM*, SFR,
    logL_Halpha) from a combined .txt file and analyze their distribution.

    The .txt file is produced by convert_properties_to_txt.py from a
    directory of per-sample property .npy files - use that script once
    to regenerate it if the underlying properties change.
    """
    print(f"\nLoading training parameters from {properties_txt}")

    if not Path(properties_txt).exists():
        raise FileNotFoundError(
            f"Properties file not found: {properties_txt}. Run convert_properties_to_txt.py "
            f"to generate it from a directory of per-sample property .npy files.")

    all_params = np.loadtxt(properties_txt)

    # Handle NaN in 4th parameter - replace with 0
    nan_rows = np.isnan(all_params[:, 3])
    all_params[nan_rows, 3] = 0.0

    print(f"Loaded {len(all_params)} parameter rows")

    # Calculate statistics
    print("\n" + "="*60)
    print("PARAMETER DISTRIBUTION ANALYSIS")
    print("="*60)

    param_mins = np.nanmin(all_params, axis=0)
    param_maxs = np.nanmax(all_params, axis=0)
    param_means = np.nanmean(all_params, axis=0)
    param_stds = np.nanstd(all_params, axis=0)

    for i in range(all_params.shape[1]):
        print(f"\nParameter {i+1}:")
        print(f"  Min:    {param_mins[i]:.6f}")
        print(f"  Max:    {param_maxs[i]:.6f}")
        print(f"  Mean:   {param_means[i]:.6f}")
        print(f"  Std:    {param_stds[i]:.6f}")

        # Check for NaN values
        nan_count = np.isnan(all_params[:, i]).sum()
        if nan_count > 0:
            print(f"  NaN values: {nan_count} ({100*nan_count/len(all_params):.2f}%)")

    print("="*60 + "\n")

    return param_mins, param_maxs, all_params

def parse_param_stats_from_log(log_path):
    """Extract the exact (param_mins, param_maxs) that conditional_dataset_npy.py
    printed at training time from a slurm-*.out log.

    conditional_dataset_npy.py._calculate_parameter_stats() redraws an unseeded
    random 1000-file sample every time training resumes, so the normalization
    bounds actually baked into a checkpoint's weights are NOT reproduced by
    recomputing min/max over --properties_txt (that recomputes from all rows,
    while training only ever normalized against whatever ~1000-file sample was
    drawn for that particular resume segment - and that draw changes every
    resume). The only way to recover the bounds a specific checkpoint was really
    trained under is to read them back out of that resume's training log. Pass
    the slurm log covering the epoch range your --epoch_id falls in.
    """
    with open(log_path) as f:
        lines = f.readlines()

    header_idxs = [i for i, l in enumerate(lines) if l.strip() == 'Parameter statistics:']
    if not header_idxs:
        raise ValueError(f"No 'Parameter statistics:' block found in {log_path}")

    # Use the last block printed in this log (closest to the end of this resume's training).
    start = header_idxs[-1] + 1
    pat = re.compile(r'Parameter \d+: \[([\-\d.eE]+), ([\-\d.eE]+)\] \(range: [\-\d.eE]+\)')
    mins, maxs = [], []
    for line in lines[start:start + 4]:
        m = pat.search(line)
        if not m:
            raise ValueError(f"Malformed parameter statistics block in {log_path}")
        mins.append(float(m.group(1)))
        maxs.append(float(m.group(2)))

    return np.array(mins), np.array(maxs)

def denormalize_parameters(normalized_params, param_mins, param_maxs):
    """Convert normalized [0,1] parameters back to original scale"""
    param_ranges = param_maxs - param_mins
    denormalized = normalized_params * param_ranges + param_mins
    return denormalized

def apply_percent_variation(base_params, variation_percent, param_mins, param_maxs, vary_mask=None):
    """Randomly vary each parameter by up to +/- variation_percent%% (independently,
    not a uniform shrink), then clip to the training set's per-parameter [min, max]
    so varied values never leave the range seen during training.

    base_params: array of shape (4,) or (N, 4)
    vary_mask: optional boolean array of shape (4,) selecting which parameters to
        vary; parameters with False are left untouched (still clipped, which is a
        no-op since they're real training values already in range).
    """
    base_params = np.asarray(base_params, dtype=np.float64)
    if vary_mask is None:
        vary_mask = np.ones(base_params.shape[-1], dtype=bool)

    if variation_percent <= 0:
        return base_params.copy()

    rand_factors = 1.0 + np.random.uniform(-variation_percent / 100.0,
                                            variation_percent / 100.0,
                                            size=base_params.shape)
    factors = np.where(vary_mask, rand_factors, 1.0)
    varied = base_params * factors
    varied = np.clip(varied, param_mins, param_maxs)
    return varied

def save_image_numpy(image_tensor, image_idx, output_dir):
    """Save the continuum channel (channel 1) only, as a (H, W) numpy array.

    The generator itself still produces both channels (channel 0 = H-alpha,
    channel 1 = continuum); channel 0 is discarded here and never written out.
    """
    img_np = image_tensor.cpu().numpy()

    if img_np.shape[0] != 2:
        print(f"WARNING: Expected 2 channels from the generator, got shape {img_np.shape}")

    continuum = img_np[1]

    # Save as numpy
    img_path = os.path.join(output_dir, "images", f"image_{image_idx}.npy")
    np.save(img_path, continuum)

    # Print channel statistics for debugging
    if image_idx < 5:  # Only for first few images
        print(f"\nImage {image_idx} continuum statistics:")
        print(f"  min={continuum.min():.4f}, max={continuum.max():.4f}, mean={continuum.mean():.4f}")

    return img_path

def save_property_numpy(params, property_idx, output_dir):
    """Save property as numpy array"""
    prop_path = os.path.join(output_dir, "properties", f"property_{property_idx}.npy")
    np.save(prop_path, params)

    return prop_path

def visualize_2channel_sample(image_tensor, params, output_dir, idx):
    """Single-panel visualization of the continuum channel (channel 1), gray_r auto-scaled."""
    img_np = image_tensor.cpu().numpy() if hasattr(image_tensor, 'cpu') else image_tensor

    param_names = [r'$z$', r'$\log M_\star$', r'SFR', r'$\log L_{H\alpha}$']
    param_str = ',  '.join(f'{n} = {v:.3f}' for n, v in zip(param_names, params))

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    im = ax.imshow(img_np[1], cmap='gray_r')
    ax.set_title('Continuum', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(param_str, fontsize=10, y=1.02)
    plt.tight_layout()

    base = os.path.join(output_dir, "visualizations", f"viz_{idx}")
    plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base + '.png'


def generate_test_samples(args):
    print("Starting 4-parameter conditional 2-channel test generation...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Verify num_channels is set correctly
    print(f"\n{'='*60}")
    print(f"CONFIGURATION CHECK")
    print(f"{'='*60}")
    print(f"Number of channels: {args.num_channels}")
    print(f"Image size: {args.image_size}")
    print(f"Number of timesteps: {args.num_timesteps}")
    print(f"{'='*60}\n")

    if args.num_channels != 2:
        print(f"WARNING: num_channels is set to {args.num_channels}, but should be 2 for 2-channel generation!")
        print("This will cause shape mismatches. Please set --num_channels 2")
        return

    # Load training parameters (physical units) and their distribution. These
    # phys_mins/phys_maxs are used ONLY for clipping +/- variation to the
    # observed training range (apply_percent_variation) and for display - they
    # are NOT necessarily what the model was actually conditioned on; see below.
    phys_mins, phys_maxs, all_train_params = load_training_parameters(args.properties_txt)

    # Ensure 4th parameter handling (replace NaN with 0)
    phys_mins = np.where(np.isnan(phys_mins), 0.0, phys_mins)
    phys_maxs = np.where(np.isnan(phys_maxs), 0.0, phys_maxs)

    # Determine the bounds actually used to turn a physical parameter row into
    # the model's conditioning input (normalize_fn below). Recomputing from
    # --properties_txt (phys_mins/phys_maxs above) does NOT reproduce what a
    # given checkpoint was trained under - see conditional_dataset_npy.py /
    # param_normalization.py for why. Priority: --stats_path (new checkpoints
    # trained with compute_param_stats.py + log-SFR transform) > --stats_log
    # (old checkpoints - recovers the exact, non-transformed bounds a specific
    # resume segment used) > explicit --param_mins/--param_maxs > fall back to
    # phys_mins/phys_maxs (old default behavior, linear, recomputed each run).
    using_transformed_stats = False
    if args.stats_path is not None:
        norm_mins, norm_maxs = pn_load_stats(args.stats_path)
        using_transformed_stats = True
        print(f"\nUsing precomputed normalization stats from {args.stats_path} "
              f"(log-SFR transformed space):")
        for i, name in enumerate(PARAM_NAMES):
            print(f"  {name}: [{norm_mins[i]:.4f}, {norm_maxs[i]:.4f}]")
    elif args.stats_log is not None:
        norm_mins, norm_maxs = parse_param_stats_from_log(args.stats_log)
        print(f"\nOverriding normalization bounds from {args.stats_log} (linear, no log-SFR "
              f"transform - use --stats_path instead for checkpoints trained with "
              f"compute_param_stats.py):")
        for i, name in enumerate(PARAM_NAMES):
            print(f"  {name}: [{phys_mins[i]:.4f}, {phys_maxs[i]:.4f}] (recomputed) "
                  f"-> [{norm_mins[i]:.4f}, {norm_maxs[i]:.4f}] (from log)")
    elif args.param_mins is not None and args.param_maxs is not None:
        norm_mins = np.array(args.param_mins)
        norm_maxs = np.array(args.param_maxs)
        print(f"\nOverriding normalization bounds with explicit --param_mins/--param_maxs "
              f"(linear, no log-SFR transform): mins={norm_mins}, maxs={norm_maxs}")
    else:
        norm_mins, norm_maxs = phys_mins, phys_maxs

    if using_transformed_stats:
        def normalize_fn(raw_params):
            return pn_normalize(raw_params, norm_mins, norm_maxs)
    else:
        norm_ranges = np.where((norm_maxs - norm_mins) == 0, 1.0, norm_maxs - norm_mins)
        def normalize_fn(raw_params):
            return (raw_params - norm_mins) / norm_ranges

    # Create output directories
    output_base = args.output_dir
    os.makedirs(os.path.join(output_base, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "properties"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "visualizations"), exist_ok=True)
    print(f"Output directory: {output_base}")

    # Load the generator model
    netG = NCSNpp(args).to(device)
    ckpt_path = f'./saved_info/dd_gan/{args.dataset}/{args.exp}/netG_{args.epoch_id}.pth'
    print(f"Loading model from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle loading from DDP state dict
    if all(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k[7:]: v for k, v in ckpt.items()}

    netG.load_state_dict(ckpt)
    netG.eval()
    print("Model loaded successfully")

    # Get diffusion parameters
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    # Use existing training parameters with optional variation
    num_train_samples = len(all_train_params)

    if args.num_samples > num_train_samples:
        print(f"Warning: Requested {args.num_samples} samples but only {num_train_samples} training samples available.")
        print(f"Will generate {num_train_samples} samples.")
        num_samples = num_train_samples
    else:
        num_samples = args.num_samples

    print(f"\n{'='*60}")
    print(f"Using existing training parameters")
    print(f"Variation: {args.variation_percent}%")
    print(f"{'='*60}\n")

    # Randomly select samples from training data (or use first N samples)
    np.random.seed(args.seed)
    if args.use_sequential:
        # Use first N samples sequentially
        selected_indices = np.arange(num_samples)
        print(f"Using first {num_samples} training samples sequentially")
    else:
        # Randomly select N samples
        selected_indices = np.random.choice(num_train_samples, size=num_samples, replace=False)
        print(f"Randomly selected {num_samples} training samples")

    base_params = all_train_params[selected_indices]

    # Randomly vary each parameter independently by up to +/- variation_percent%,
    # then clip to the training set's per-parameter [min, max] (physical units)
    # so varied values never fall outside the range seen during training.
    test_params_original = apply_percent_variation(base_params, args.variation_percent,
                                                     phys_mins, phys_maxs)

    print(f"\nApplied random +/-{args.variation_percent}% variation per parameter "
          f"(each clipped to its training min/max)")

    # Turn physical-unit parameters into the model's conditioning input.
    test_params_normalized = normalize_fn(test_params_original)

    # Flag anything landing outside [0,1]: with the correct bounds this means
    # the raw value truly falls outside what this checkpoint was conditioned
    # on, i.e. the generator is being asked to extrapolate.
    out_of_range = (test_params_normalized < 0) | (test_params_normalized > 1)
    if out_of_range.any():
        n_bad = out_of_range.any(axis=1).sum()
        print(f"WARNING: {n_bad}/{len(test_params_normalized)} samples have at least one "
              f"parameter normalized outside [0,1] given the bounds in use — the generator "
              f"is being asked to extrapolate beyond what it was conditioned on for this checkpoint.")

    print(f"Generating {num_samples} test samples")

    # Save parameter information
    normalization_source = (
        f"{args.stats_path} (log-SFR transformed)" if args.stats_path is not None else
        f"{args.stats_log} (linear)" if args.stats_log is not None else
        "explicit --param_mins/--param_maxs (linear)" if args.param_mins is not None else
        f"recomputed from --properties_txt ({args.properties_txt}) (linear)"
    )
    param_info = {
        "num_channels": args.num_channels,
        "image_size": args.image_size,
        "variation_percent": args.variation_percent,
        "selection_mode": "sequential" if args.use_sequential else "random",
        "normalization_source": normalization_source,
        "original_ranges": {
            "param1": [float(phys_mins[0]), float(phys_maxs[0])],
            "param2": [float(phys_mins[1]), float(phys_maxs[1])],
            "param3": [float(phys_mins[2]), float(phys_maxs[2])],
            "param4": [float(phys_mins[3]), float(phys_maxs[3])]
        },
        "num_training_samples": len(all_train_params),
        "num_generated_samples": num_samples,
        "test_samples": []
    }

    # Generate samples one by one
    print("\nGenerating samples...")
    for idx in tqdm(range(num_samples)):
        # Get normalized parameters for this sample
        norm_params = test_params_normalized[idx]

        # Convert to tensor - ensure shape is (1, 4)
        param_tensor = torch.tensor([norm_params], device=device).float()

        # Verify parameter tensor shape
        if param_tensor.shape[1] != 4:
            print(f"ERROR: Parameter tensor has wrong shape: {param_tensor.shape}, expected (1, 4)")
            continue

        # Get original scale parameters
        orig_params = test_params_original[idx]

        # Initialize with random noise - CRITICAL: must be 2 channels!
        x_t_1 = torch.randn(1, args.num_channels, args.image_size, args.image_size).to(device)

        # Generate sample
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args, param_tensor)

        # Denormalize [-1, 1] → [0, 1] to match original data range
        fake_sample = (fake_sample + 1.0) / 2.0
        fake_sample = fake_sample.clamp(0.0, 1.0)

        # Verify output shape
        expected_shape = (1, args.num_channels, args.image_size, args.image_size)
        if fake_sample.shape != expected_shape:
            print(f"ERROR: Generated sample has wrong shape: {fake_sample.shape}, expected {expected_shape}")
            continue

        # Print statistics for first few samples
        if idx < 5:
            print(f"\nSample {idx}:")
            print(f"  Shape: {fake_sample.shape}")
            print(f"  Value range: [{fake_sample.min().item():.4f}, {fake_sample.max().item():.4f}]")
            print(f"  Parameters (normalized): {norm_params}")
            print(f"  Parameters (original): {orig_params}")

        # Visualize the first num_visualize samples immediately (before continuing generation)
        if idx < args.num_visualize:
            viz_path = visualize_2channel_sample(fake_sample[0], orig_params, output_base, idx)
            print(f"  Saved → {viz_path}")

        # Save continuum channel (channel 1) as numpy (shape: H, W); channel 0 (H-alpha) is discarded
        img_path = save_image_numpy(fake_sample[0], idx, output_base)

        # Save properties as numpy (original scale)
        prop_path = save_property_numpy(orig_params, idx, output_base)

        # Save to parameter info (every 100 samples to avoid huge JSON)
        if idx % 100 == 0 or idx < 10 or idx >= num_samples - 10:
            param_info["test_samples"].append({
                "index": idx,
                "training_index": int(selected_indices[idx]),
                "normalized_params": norm_params.tolist(),
                "original_params": orig_params.tolist(),
                "image_file": f"images/image_{idx}.npy",
                "property_file": f"properties/property_{idx}.npy"
            })

    # Save parameter info to JSON
    with open(os.path.join(output_base, "test_info.json"), 'w') as f:
        json.dump(param_info, f, indent=2)

    print(f"\n✓ Successfully generated {num_samples} continuum samples")
    print(f"✓ Images saved to {output_base}/images/ (shape: {args.image_size}, {args.image_size}; channel 0/H-alpha discarded)")
    print(f"✓ Properties saved to {output_base}/properties/")
    print(f"✓ Visualizations saved to {output_base}/visualizations/ (first {min(args.num_visualize, num_samples)} samples)")
    print(f"✓ Test information saved to test_info.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDGAN 4-Parameter 2-Channel Test Generation')

    # Model parameters (should match training)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--epoch_id', type=int, default=300,
                        help='which checkpoint to use')
    parser.add_argument('--num_channels', type=int, default=2,  # FIXED: Changed from 1 to 2
                        help='channel of image (MUST be 2 for 2-channel generation)')
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)

    parser.add_argument('--num_channels_dae', type=int, default=128)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # Experiment settings
    parser.add_argument('--exp', default='experiment_npy_18k',
                        help='experiment name (should match training)')
    parser.add_argument('--dataset', default='custom_conditional',
                        help='dataset name (should match training)')
    parser.add_argument('--image_size', type=int, default=128,  # FIXED: Changed from 32 to 128
                        help='image size during training')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--param_emb_dim', type=int, default=128)

    # Input/Output settings
    parser.add_argument('--properties_txt', type=str, default='properties_grs_cut.txt',
                        help='combined .txt file of training properties (one row per sample: '
                             'z, logM*, SFR, logL_Halpha), produced by convert_properties_to_txt.py')
    parser.add_argument('--output_dir', type=str, default='../GRS/generated',
                        help='directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='number of samples to generate')

    # Normalization-bounds override (see param_normalization.py for why this matters:
    # --properties_txt alone does not reproduce what a given checkpoint was actually
    # trained under, since training previously resampled its stats unseeded on every
    # resume). Priority: --stats_path > --stats_log > --param_mins/--param_maxs.
    parser.add_argument('--stats_path', type=str, default=None,
                        help='path to a precomputed param_stats .npz from compute_param_stats.py '
                             '(log-SFR transformed space). Use this for any checkpoint trained '
                             'with the stats_path-enabled conditional_dataset_npy.py.')
    parser.add_argument('--stats_log', type=str, default=None,
                        help='[legacy, pre-log-transform checkpoints only] path to the '
                             'slurm-*.out training log covering the resume segment that '
                             'produced --epoch_id; its printed "Parameter statistics" block '
                             '(linear, no log-SFR transform) is used as the normalization '
                             'bounds instead of recomputing from --properties_txt. Ignored if '
                             '--stats_path is given.')
    parser.add_argument('--param_mins', type=float, nargs=4, default=None,
                        help='[legacy, linear] explicit override for param_mins (4 values: z, '
                             'logM*, SFR, logL_Halpha), used instead of recomputing from '
                             '--properties_txt. Ignored if --stats_path or --stats_log is given.')
    parser.add_argument('--param_maxs', type=float, nargs=4, default=None,
                        help='[legacy, linear] explicit override for param_maxs (4 values), '
                             'paired with --param_mins. Ignored if --stats_path or --stats_log '
                             'is given.')

    # Variation settings
    parser.add_argument('--variation_percent', type=float, default=0.0,
                        help='each parameter of each selected training sample is randomly varied by '
                             'up to +/- this percent (independently, not a uniform shrink), then '
                             'clipped to that parameter\'s [min, max] over the whole training set')
    parser.add_argument('--use_sequential', action='store_true', default=False,
                        help='if set, use first N training samples sequentially instead of random selection')
    parser.add_argument('--num_visualize', type=int, default=20,
                        help='number of generated samples (from the start of the run) to also save as '
                             'PNG visualizations, in addition to the .npy saved for every sample')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generate_test_samples(args)
