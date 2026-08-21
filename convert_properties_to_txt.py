#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combine a directory of per-sample property .npy files (one file per
training cube, each holding [z, logM*, SFR, logL_Halpha]) into a single
.txt file.

Run this once against GRS_cut_snap_43_removed/properties (or any similar
properties directory) to produce a compact file that test_generation_2_channel.py
can load its conditioning parameters from, instead of requiring the full
directory of 60k+ individual .npy files. This lets that directory be removed
before pushing to GitHub.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

PARAM_NAMES = ['z', 'logM_star', 'SFR', 'logL_Halpha']


def convert(properties_dir, output_path):
    property_files = sorted(Path(properties_dir).glob("*.npy"))
    if len(property_files) == 0:
        raise FileNotFoundError(f"No property files found in {properties_dir}")

    print(f"Found {len(property_files)} property files in {properties_dir}")

    all_params = []
    for prop_file in tqdm(property_files, desc="Loading properties"):
        params = np.load(prop_file)
        # Same NaN handling as test_generation_2_channel.py: replace NaN
        # 4th parameter (logL_Halpha) with 0.
        if len(params) >= 4 and np.isnan(params[3]):
            params[3] = 0.0
        all_params.append(params)

    all_params = np.array(all_params, dtype=np.float64)

    header = f"{len(all_params)} rows x {all_params.shape[1]} cols: " + " ".join(PARAM_NAMES)
    np.savetxt(output_path, all_params, fmt="%.8e", header=header)

    print(f"\nSaved {all_params.shape[0]} rows x {all_params.shape[1]} cols -> {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1e6:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine per-sample property .npy files into one .txt file")
    parser.add_argument('--properties_dir', type=str, default='GRS_cut_snap_43_removed/properties',
                        help='directory containing one .npy file per sample (4 params each)')
    parser.add_argument('--output', type=str, default='properties_grs_cut.txt',
                        help='path to write the combined .txt file to')
    args = parser.parse_args()

    convert(args.properties_dir, args.output)
