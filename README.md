# H-alpha + continuum sample generation

`test_generation_2_channel.py` loads a trained DDGAN checkpoint (a conditional
2-channel generator: channel 0 = H-alpha, channel 1 = continuum) and generates
`N` new samples. **Both channels are saved** for every sample.

For each generated sample the script writes:

- `images/image_<idx>.npy` — both channels, shape `(2, image_size, image_size)`
  (`[0]` = H-alpha, `[1]` = continuum), values in `[0, 1]`.
- `properties/property_<idx>.npy` — the 4 physical-unit conditioning parameters
  `[z, logM*, SFR, logL_Halpha]` used to generate that sample.
- `test_info.json` — run configuration + a sampled record of indices, parameters,
  and file paths (not every sample, to keep the file small).

In addition, the **first `--num_visualize` samples** (default 20) also get a PNG
visualization in `visualizations/`:

- `viz_<idx>.png` — H-alpha and continuum side by side, linear gray scale.

## Requirements

- Python 3.10, PyTorch 2.x + torchvision (CUDA build recommended but CPU works).
- `numpy`, `matplotlib`, `tqdm`.
- [Git LFS](https://git-lfs.com/) to fetch the checkpoint (`git lfs install` once per machine, then a normal `git clone`/`git pull` fetches it automatically).
- A trained checkpoint under `saved_info/dd_gan/<dataset>/<exp>/netG_<epoch_id>.pth`. Only
  `netG_70.pth` (under `experiment_latest_hasti_60k_high_res_snap43_removed`) is the supported
  checkpoint for this repo (~569MB, intended to be tracked via Git LFS); other checkpoints are
  kept local only and not pushed.
- A combined properties `.txt` file (see below), used to determine each
  parameter's valid `[min, max]` range and as the normalization-bounds
  fallback.
- A parameter CSV file (see below) specifying which conditioning values to
  actually generate images for.

## Properties file

`properties_highres_cut_snap43_removed.txt` (for `netG_70.pth`; built with
`convert_properties_to_txt.py --properties_dir <processed_new_highres_halpha_cut_snap43_removed/properties> --output properties_highres_cut_snap43_removed.txt`;
the older `properties_grs_cut.txt` belongs to `netG_122.pth`) (one row per training sample: `z logM* SFR
logL_Halpha`) is the training set's full parameter distribution, used by
`test_generation_2_channel.py` (via `--properties_txt`, default
`properties_highres_cut_snap43_removed.txt`) only to compute each parameter's observed `[min,
max]` — i.e. what `param.csv` values are validated against — and as the
normalization-bounds fallback when `--stats_path` isn't given. It is no longer
where conditioning values themselves come from; that's `param.csv` now.

```bash
python3 -c "
import numpy as np
a = np.loadtxt('properties_highres_cut_snap43_removed.txt')
print(a.shape)   # (60305, 4)
print(a[0])       # first row: z, logM*, SFR, logL_Halpha
"
```

## Parameter CSV (`param.csv`)

Conditioning values to generate images for are read from a CSV file
(`--params_csv`, default `param.csv`) — **one row per sample to generate**.

- First row is a header, one column per parameter, in order `z, logM*, SFR,
  logL_Halpha`, formatted as `<name> [<units>]`. Parameters that are
  themselves log quantities (`logM*`, `logL_Halpha`) carry "log" in the name;
  `SFR` is stored/entered as linear (Msun/yr) — the model log10-transforms it
  internally at normalization time (see `param_normalization.py`), so don't
  pre-log it yourself.
- Every value must lie within that parameter's `[min, max]` as observed in
  `--properties_txt` — `test_generation_2_channel.py` validates this on load
  and raises a `ValueError` naming the offending row/column if a value falls
  outside that range (asking the generator to condition on an out-of-range
  value means extrapolating beyond what it was trained on).

Example (`param.csv` — 5 rows sampled at random (seed 42) from the high-res training
properties, `processed_new_highres_halpha_cut_snap43_removed/properties`):

```csv
z [redshift],logM* [log(Msun)],SFR [Msun/yr],logL_Halpha [log(Lsun)]
0.95,9.29,1.30,7.98
1.50,8.38,0.22,7.48
1.04,8.49,0.16,7.14
1.36,9.33,1.28,8.00
1.11,9.27,0.96,7.83
```

## Usage

Run from the repo root (`DDGAN_GRS_inference_pipeline/`):

```bash
python test_generation_2_channel.py \
  --exp experiment_latest_hasti_60k_high_res_snap43_removed \
  --dataset custom_conditional \
  --epoch_id 70 \
  --num_channels 2 \
  --image_size 256 \
  --ch_mult 1 1 2 2 4 4 \
  --properties_txt properties_highres_cut_snap43_removed.txt \
  --stats_path param_stats_highres_cut_snap43_removed.npz \
  --params_csv param.csv \
  --output_dir generated_continuum \
  --variation_percent 0 \
  --num_visualize 20
```

This generates one 2-channel sample per row of `param.csv` into
`generated_continuum/`, with the first `--num_visualize` also rendered as
PNGs. `--image_size`, `--ch_mult`, and `--epoch_id` must match whatever
architecture/checkpoint you're loading — the values above are verified
working for `netG_70.pth` under
`experiment_latest_hasti_60k_high_res_snap43_removed`. Other checkpoints/experiments
may need different `--ch_mult`/`--num_channels_dae` values and will fail to load with a
`size mismatch` / missing-key error otherwise.

The normalization stats must also match the checkpoint: `netG_70.pth` was trained with
`param_stats_highres_cut_snap43_removed.npz` (log-SFR transformed; logL_Halpha range
≈ 3.53–10.31). The older `param_stats_grs_cut.npz` belongs to the previous
`netG_122.pth` checkpoint and will give wrongly-scaled conditioning with `netG_70.pth`.

> **Note:** `--properties_txt` now defaults to `properties_highres_cut_snap43_removed.txt`
> (60305 rows, logL_Halpha ≈ 3.53–10.31), matching `param_stats_highres_cut_snap43_removed.npz`.
> The older `properties_grs_cut.txt` (logL_Halpha max ≈ 8.33) would reject `param.csv` values
> above that range.

### Key arguments

| Argument | Purpose |
|---|---|
| `--params_csv` | CSV of conditioning parameters to generate, one row per sample (see above). Default `param.csv`. |
| `--num_samples` | Optional cap on how many rows of `--params_csv` to use. Defaults to using every row. |
| `--num_visualize` | How many of the generated samples (from the start of the run) also get a PNG in `visualizations/`. Every sample still gets a `.npy` regardless of this setting. |
| `--exp`, `--dataset`, `--epoch_id` | Select which checkpoint to load, from `saved_info/dd_gan/<dataset>/<exp>/netG_<epoch_id>.pth`. |
| `--properties_txt` | Combined `.txt` file of training conditioning parameters (see above); used to determine each parameter's valid `[min, max]` range and as the normalization-bounds fallback. |
| `--stats_path` | Precomputed normalization stats (`.npz` from `compute_param_stats.py`, log-SFR transformed). Use this if the checkpoint was trained with that normalization — required for `netG_70.pth` (use `param_stats_highres_cut_snap43_removed.npz`). |
| `--stats_log` / `--param_mins` / `--param_maxs` | Legacy alternatives to `--stats_path` for older checkpoints — see the docstrings in the script for when each applies. |
| `--variation_percent` | Randomly perturb each conditioning parameter from `param.csv` by up to +/- this percent (clipped to the training set's min/max) before generating. `0` (default) uses the CSV values unmodified. |
| `--image_size`, `--num_timesteps`, model architecture flags | Must match the values used at training time for the chosen checkpoint. |

Run `python test_generation_2_channel.py --help` for the full list of options.

## Output layout

```
generated_continuum/
├── images/
│   ├── image_0.npy          # H-alpha + continuum, shape (2, image_size, image_size)
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

img = np.load("generated_continuum/images/image_0.npy")   # (2, image_size, image_size)
halpha, continuum = img[0], img[1]
params = np.load("generated_continuum/properties/property_0.npy")  # [z, logM*, SFR, logL_Halpha]
```
