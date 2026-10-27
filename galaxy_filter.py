"""Post-generation filter that rejects unrealistic DDGAN galaxies.

In sparsely sampled corners of parameter space the generator can produce
galaxies that do not look like real galaxies with the requested parameters.
Example: z = 1.5, logM* = 11.57, SFR = 1.3, logL_Halpha = 6.66 (a massive,
quiescent galaxy; only ~6 of the 60,305 training galaxies resemble it). Real
galaxies there are smooth ellipticals with almost no H-alpha, but the
generator returns a patchy continuum envelope that runs to the image edges
and a spurious ring of H-alpha clumps.

The filter asks whether a generated image looks like a galaxy with the
requested parameters. A 4-member DualBranchDenseNet regression ensemble,
trained on the real TNG-50 training images, predicts logM*, log SFR and
logL_Halpha from the image (z is not used: it cannot be recovered even from
real images, R^2 = 0.34). The fidelity score is the RMS of the prediction
errors, each in units of the ensemble's RMSE on the real test set:

    fidelity = sqrt(mean_k ((pred_k - requested_k) / RMSE_k)^2),  k = logM*, log SFR, logL_Halpha

A real test image scores ~1.5 on average. Images with fidelity > threshold
(default 4.0) are rejected. Measured on netG_104:
  - all 16 samples at the parameters above score 4.2-7.3 (rejected); the
    ensemble reads them as logM* ~ 10.5 instead of 11.6
  - 1,500 samples conditioned on random training rows: median 1.6, 99th
    percentile 3.8, so ~1% of ordinary samples are rejected
Caveat: real galaxies in the same rare corner also score high (1.8-4.7), so
the filter is strict there; it never touches real images.

Usage, on an existing output directory of test_generation_2_channel.py
(writes filter_results.csv, moves rejected samples to <gen_dir>/rejected/):
    python galaxy_filter.py --gen_dir generated_continuum
Or during generation: test_generation_2_channel.py --filter [--filter_retries N].

Needs the ensemble checkpoints densenet_member{0..3}_seed{41,42,100,137}.pt in
--model_dir (default filter_models/), copied from
checkpoints_ensemble_highres_snap43_removed/ of the training repo.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

ENSEMBLE_SEEDS = [41, 42, 100, 137]
N_Z_CLASSES = 14

# Normalization of the ensemble's outputs (checkpoints_ensemble_highres_snap43_removed/
# param_stats.npy): [z, logM*, log10 SFR, logL_Halpha]
PRED_MINS = np.array([0.8899999856948853, 8.000192642211914, -0.9999739527702332, 3.5334067344665527])
PRED_MAXS = np.array([1.5, 12.248202323913574, 2.8396565914154053, 10.314964294433594])
# Ensemble RMSE on the 6,030 real test images (test_eval/test_predictions.npz, rmse_log)
RMSE_LOG = np.array([0.15299381, 0.1224847, 0.13678256, 0.12171864])
SCORED = [1, 2, 3]  # logM*, log SFR, logL_Halpha
DEFAULT_THRESHOLD = 4.0


# Model: must match train_regression_highres.py / eval_generated_regression.py exactly.
def _densenet_backbone(in_channels):
    net = tv_models.densenet121(weights=None)
    net.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = nn.Sequential(net.features, nn.ReLU(inplace=True),
                             nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    return backbone, net.classifier.in_features


class DualBranchDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_full, feat_dim = _densenet_backbone(in_channels=2)
        self.backbone_ha, _ = _densenet_backbone(in_channels=1)
        self.head_other = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(256, 3))
        self.head_z_cls = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(256, N_Z_CLASSES))
        self.head_ha = nn.Sequential(nn.Linear(feat_dim + 1, 128), nn.ReLU(inplace=True),
                                     nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        feat_full = self.backbone_full(x)
        feat_ha = self.backbone_ha(x[:, :1])
        ha_cover = x[:, 0].mean(dim=[1, 2]).unsqueeze(-1)
        return torch.cat([self.head_other(feat_full),
                          self.head_ha(torch.cat([feat_ha, ha_cover], dim=-1))], dim=1)


class GalaxyFilter:
    def __init__(self, model_dir="filter_models", device=None, threshold=DEFAULT_THRESHOLD):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.models = []
        for i, seed in enumerate(ENSEMBLE_SEEDS):
            path = Path(model_dir) / f"densenet_member{i}_seed{seed}.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; copy the regression ensemble checkpoints "
                                        f"into {model_dir}/ (see the galaxy_filter.py docstring)")
            m = DualBranchDenseNet().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            self.models.append(m.eval())

    def predict(self, imgs):
        """imgs (N, 2, H, W) in [0, 1] -> predicted [z, logM*, log10 SFR, logL_Halpha] (N, 4)."""
        x = torch.from_numpy(np.clip(np.asarray(imgs, dtype=np.float32), 0.0, 1.0)).to(self.device)
        with torch.no_grad():
            p = torch.stack([m(x) for m in self.models]).mean(0).cpu().numpy()
        return p * (PRED_MAXS - PRED_MINS) + PRED_MINS

    def score(self, imgs, params):
        """imgs (N, 2, H, W); params (N, 4) or (4,) requested [z, logM*, SFR (linear), logL_Halpha].
        Returns (fidelity (N,), predicted (N, 4) with SFR as log10)."""
        pred = self.predict(imgs)
        target = np.array(np.broadcast_to(params, pred.shape), dtype=np.float64)
        target[:, 2] = np.log10(np.maximum(target[:, 2], 1e-6))
        err = (pred[:, SCORED] - target[:, SCORED]) / RMSE_LOG[SCORED]
        return np.sqrt((err ** 2).mean(1)), pred

    def passes(self, imgs, params):
        """Boolean mask (N,): True for images that look like their requested parameters."""
        return self.score(imgs, params)[0] <= self.threshold


def filter_directory(gen_dir, model_dir="filter_models", threshold=DEFAULT_THRESHOLD,
                     move_rejected=True, batch_size=16):
    """Score every images/image_<i>.npy in a test_generation_2_channel.py output
    directory against properties/property_<i>.npy, write filter_results.csv and
    (optionally) move rejected samples, with their properties and visualizations,
    to <gen_dir>/rejected/."""
    gen_dir = Path(gen_dir)
    gf = GalaxyFilter(model_dir, threshold=threshold)
    idxs = sorted(int(p.stem.split("_")[1]) for p in (gen_dir / "images").glob("image_*.npy"))
    rows = []
    for start in range(0, len(idxs), batch_size):
        batch = idxs[start:start + batch_size]
        imgs = np.stack([np.load(gen_dir / "images" / f"image_{i}.npy") for i in batch])
        params = np.stack([np.load(gen_dir / "properties" / f"property_{i}.npy").ravel() for i in batch])
        fid, pred = gf.score(imgs, params)
        for i, p, f, pr in zip(batch, params, fid, pred):
            rows.append([i, *p, *pr[1:], f, int(f <= threshold)])

    with open(gen_dir / "filter_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "z", "logM*", "SFR", "logL_Halpha",
                    "pred_logM*", "pred_log10_SFR", "pred_logL_Halpha", "fidelity", "passed"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:-1]] + [r[-1]])

    rejected = [r[0] for r in rows if not r[-1]]
    if move_rejected and rejected:
        for sub in ("images", "properties", "visualizations"):
            os.makedirs(gen_dir / "rejected" / sub, exist_ok=True)
        for i in rejected:
            for sub, name in (("images", f"image_{i}.npy"), ("properties", f"property_{i}.npy"),
                              ("visualizations", f"viz_{i}.png")):
                src = gen_dir / sub / name
                if src.exists():
                    shutil.move(str(src), str(gen_dir / "rejected" / sub / name))
    print(f"Filter (fidelity <= {threshold}): {len(rows) - len(rejected)}/{len(rows)} passed, "
          f"{len(rejected)} rejected" + (f" -> {gen_dir / 'rejected'}" if move_rejected and rejected else ""))
    print(f"Scores written to {gen_dir / 'filter_results.csv'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reject generated galaxies that do not look like their requested parameters")
    parser.add_argument("--gen_dir", required=True, help="output directory of test_generation_2_channel.py")
    parser.add_argument("--model_dir", default="filter_models", help="regression ensemble checkpoints")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="maximum fidelity score to keep an image (real test images average ~1.5)")
    parser.add_argument("--keep_rejected_in_place", action="store_true",
                        help="only write filter_results.csv, do not move rejected samples")
    a = parser.parse_args()
    filter_directory(a.gen_dir, a.model_dir, a.threshold, move_rejected=not a.keep_rejected_in_place)
