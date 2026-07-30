"""Single source of truth for how the 4 conditioning parameters
(z, logM*, SFR, logL_Halpha) are transformed and normalized to [0,1] for the
generator's conditioning input.

SFR (index 2) spans ~4 orders of magnitude (0.1 to ~691 in GRS_cut) while the
other three parameters are already well-scaled: z has a narrow range, and
logM*/logL_Halpha are already log-transformed upstream in the properties
data. Under plain linear min-max, SFR's astrophysically-normal regime
(~0.1-10, ~90% of the data) gets crushed into the bottom ~1.5% of [0,1],
leaving almost no resolution for the conditioning network to distinguish
typical galaxies. Log10-transforming SFR before min-max fixes that.

Both training (conditional_dataset_npy.py) and inference
(test_generation_2_channel.py, generate_real_vs_generated.py) import this
module so the transform and the normalization bounds can never drift apart
between the two. Previously each side recomputed its own stats independently
(and training resampled a random subset, unseeded, on every job resume) -
that's what produced the checkpoint-vs-inference mismatches this file exists
to prevent.
"""
import numpy as np

PARAM_NAMES = ['z', 'logM*', 'SFR', 'logL_Halpha']
LOG_PARAM_INDICES = [2]  # SFR only; the other three are already well-scaled

# GRS_cut/properties has no SFR<=0 entries (observed min 0.1), but the
# uncut GRS/properties does contain exact zeros - log10(0) = -inf, which
# would silently poison every downstream min/max/normalize call. Floor at a
# value well below any real observed SFR so it never affects the actual
# training range, it only guards against pathological zero/negative input.
LOG_FLOOR = 1e-3


def transform_params(raw_params):
    """raw_params: array-like (..., 4) of physical-unit values.
    Returns a copy with LOG_PARAM_INDICES log10-transformed (floored at
    LOG_FLOOR to avoid log10(0) = -inf on zero/negative values)."""
    raw_params = np.asarray(raw_params, dtype=np.float64)
    out = raw_params.copy()
    for idx in LOG_PARAM_INDICES:
        out[..., idx] = np.log10(np.maximum(out[..., idx], LOG_FLOOR))
    return out


def inverse_transform_params(transformed_params):
    """Inverse of transform_params: undoes the log10 on LOG_PARAM_INDICES,
    returning physical-unit values."""
    transformed_params = np.asarray(transformed_params, dtype=np.float64)
    out = transformed_params.copy()
    for idx in LOG_PARAM_INDICES:
        out[..., idx] = 10.0 ** out[..., idx]
    return out


def compute_stats(all_raw_params):
    """all_raw_params: array (N, 4) of physical-unit values (e.g. every row
    in a properties folder). Returns (param_mins, param_maxs) in TRANSFORMED
    space, ready to save/use for normalization."""
    transformed = transform_params(all_raw_params)
    return np.nanmin(transformed, axis=0), np.nanmax(transformed, axis=0)


def save_stats(path, param_mins, param_maxs):
    np.savez(path, param_mins=param_mins, param_maxs=param_maxs,
             log_param_indices=np.array(LOG_PARAM_INDICES))


def load_stats(path):
    """Returns (param_mins, param_maxs) in transformed space, as saved by
    save_stats / compute_param_stats.py. Raises if the file's log-transform
    convention doesn't match this module's current LOG_PARAM_INDICES, so a
    stale stats file can't silently get used with mismatched code."""
    data = np.load(path)
    saved_log_idx = list(data['log_param_indices'])
    if saved_log_idx != LOG_PARAM_INDICES:
        raise ValueError(
            f"{path} was computed with log_param_indices={saved_log_idx}, "
            f"but this code expects {LOG_PARAM_INDICES}. Recompute the stats "
            f"file with the current compute_param_stats.py."
        )
    return data['param_mins'], data['param_maxs']


def normalize(raw_params, param_mins, param_maxs):
    """raw_params: physical-unit values (..., 4). param_mins/maxs: transformed
    -space bounds (e.g. from load_stats). Returns normalized values (..., 4).
    Not clamped to [0,1] - callers should check/clip if that matters for
    their use case (e.g. flagging out-of-range requests)."""
    transformed = transform_params(raw_params)
    param_ranges = np.where((param_maxs - param_mins) == 0, 1.0, param_maxs - param_mins)
    return (transformed - param_mins) / param_ranges


def denormalize(norm_params, param_mins, param_maxs):
    """Inverse of normalize(): returns physical-unit values."""
    param_ranges = np.where((param_maxs - param_mins) == 0, 1.0, param_maxs - param_mins)
    transformed = np.asarray(norm_params, dtype=np.float64) * param_ranges + param_mins
    return inverse_transform_params(transformed)
