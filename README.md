# Continuum sample generation

`test_generation_2_channel.py` loads a trained DDGAN checkpoint (a conditional
2-channel generator: channel 0 = H-alpha, channel 1 = continuum) and generates
`N` new samples. Only the **continuum channel is saved** — channel 0 (H-alpha)
is generated internally (the model needs both) but discarded before writing
any output.

For each generated sample the script writes:

- `images/image_<idx>.npy` — the continuum channel, shape `(image_size, image_size)`.
- `properties/property_<idx>.npy` — the 4 physical-unit conditioning parameters
  `[z, logM*, SFR, logL_Halpha]` used to generate that sample.
- `test_info.json` — run configuration + a sampled record of indices, parameters,
  and file paths (not every sample, to keep the file small).

In addition, the **first `--num_visualize` samples** (default 20) also get a PNG
visualization in `visualizations/`:

- `viz_<idx>.png` — continuum, linear gray scale.

## Requirements

- Python 3.10, PyTorch 2.x + torchvision (CUDA build recommended but CPU works).
- `numpy`, `matplotlib`, `tqdm`.
- [Git LFS](https://git-lfs.com/) to fetch the checkpoint (`git lfs install` once per machine, then a normal `git clone`/`git pull` fetches it automatically).
- A trained checkpoint under `saved_info/dd_gan/<dataset>/<exp>/netG_<epoch_id>.pth`. Only
  `netG_40.pth` (under `experiment_latest_hasti_60k_corrected_snap43_removed`) is included in
  this repo, tracked via Git LFS since it's ~569MB; other checkpoints (e.g. `netG_32.pth`) are
  kept local only and not pushed.
- A combined properties `.txt` file (see below) with the 4 conditioning parameters
  per training sample, used to pick conditioning values for generation.

## Properties file

Conditioning values are drawn from the training set's parameters, stored in
`properties_grs_cut.txt` (one row per sample: `z logM* SFR logL_Halpha`) rather
than as 60k+ individual per-sample `.npy` files. `test_generation_2_channel.py`
reads conditioning parameters from this file via `--properties_txt` (default
`properties_grs_cut.txt`).

```bash
python3 -c "
import numpy as np
a = np.loadtxt('properties_grs_cut.txt')
print(a.shape)   # (60288, 4)
print(a[0])       # first row: z, logM*, SFR, logL_Halpha
"
```

## Usage

Run from the repo root (`DDGAN_GRS_inference_pipeline/`):

```bash
python test_generation_2_channel.py \
  --exp experiment_latest_hasti_60k_corrected_snap43_removed \
  --dataset custom_conditional \
  --epoch_id 40 \
  --num_channels 2 \
  --image_size 256 \
  --ch_mult 1 1 2 2 4 4 \
  --properties_txt properties_grs_cut.txt \
  --stats_path param_stats_grs_cut.npz \
  --output_dir generated_continuum \
  --variation_percent 0 \
  --num_samples 100 \
  --num_visualize 20
```

This generates 100 continuum samples into `generated_continuum/`, with the
first 20 also rendered as PNGs. `--image_size`, `--ch_mult`, and `--epoch_id`
must match whatever architecture/checkpoint you're loading — the values above
are verified working for `netG_40.pth` under
`experiment_latest_hasti_60k_corrected_snap43_removed` (the only checkpoint
included in this repo). Other checkpoints/experiments may need different
`--ch_mult`/`--num_channels_dae` values and will fail to load with a
`size mismatch` / missing-key error otherwise.

### Key arguments

| Argument | Purpose |
|---|---|
| `--num_samples` | Number of samples (`n`) to generate. |
| `--num_visualize` | How many of the generated samples (from the start of the run) also get a PNG in `visualizations/`. Every sample still gets a `.npy` regardless of this setting. |
| `--exp`, `--dataset`, `--epoch_id` | Select which checkpoint to load, from `saved_info/dd_gan/<dataset>/<exp>/netG_<epoch_id>.pth`. |
| `--properties_txt` | Combined `.txt` file of training conditioning parameters (see above); conditioning values for generation are drawn from these rows (optionally perturbed, see `--variation_percent`). |
| `--stats_path` | Precomputed normalization stats (`.npz` from `compute_param_stats.py`, log-SFR transformed). Use this if the checkpoint was trained with that normalization. |
| `--stats_log` / `--param_mins` / `--param_maxs` | Legacy alternatives to `--stats_path` for older checkpoints — see the docstrings in the script for when each applies. |
| `--variation_percent` | Randomly perturb each conditioning parameter of the selected training sample by up to +/- this percent (clipped to the training set's min/max) before generating. `0` (default) uses the training values unmodified. |
| `--use_sequential` | Use the first `N` rows of `--properties_txt` in order instead of a random subset. |
| `--image_size`, `--num_timesteps`, model architecture flags | Must match the values used at training time for the chosen checkpoint. |

Run `python test_generation_2_channel.py --help` for the full list of options.

## Output layout

```
generated_continuum/
├── images/
│   ├── image_0.npy          # continuum only, shape (image_size, image_size)
│   ├── image_1.npy
│   └── ...
├── properties/
│   ├── property_0.npy       # [z, logM*, SFR, logL_Halpha], physical units
│   └── ...
├── visualizations/
│   ├── viz_0.png
│   └── ...                  # only for the first --num_visualize samples
└── test_info.json
```

## Loading a generated sample

```python
import numpy as np

continuum = np.load("generated_continuum/images/image_0.npy")   # (image_size, image_size)
params = np.load("generated_continuum/properties/property_0.npy")  # [z, logM*, SFR, logL_Halpha]
```
