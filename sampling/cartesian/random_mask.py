import numpy as np


def random_cartesian_mask(shape, acceleration=4, center_fraction=0.08, seed=None):
    """
    Generate a 1D Cartesian undersampling mask for 2D MRI.

    The mask samples complete k-space columns/rows depending on how it is used.
    Here, we generate a phase-encoding style mask of shape (H, W).

    Parameters
    ----------
    shape : tuple
        K-space shape as (H, W).

    acceleration : int or float
        Approximate acceleration factor.
        Example: acceleration=4 means approximately 25% sampling.

    center_fraction : float
        Fraction of low-frequency central k-space lines to always keep.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    mask : np.ndarray
        Binary undersampling mask of shape (H, W), dtype float32.
    """
    rng = np.random.default_rng(seed)

    height, width = shape

    # Number of center low-frequency lines
    num_low_freqs = int(round(height * center_fraction))

    # Approximate number of total sampled phase-encoding lines
    num_total_samples = int(round(height / acceleration))

    # Number of random outer k-space lines
    num_random_samples = max(num_total_samples - num_low_freqs, 0)

    mask_1d = np.zeros(height, dtype=np.float32)

    # Always sample the center region
    center = height // 2
    low_start = center - num_low_freqs // 2
    low_end = low_start + num_low_freqs
    mask_1d[low_start:low_end] = 1.0

    # Randomly sample remaining lines outside the center
    all_indices = np.arange(height)
    center_indices = np.arange(low_start, low_end)
    candidate_indices = np.setdiff1d(all_indices, center_indices)

    if num_random_samples > 0:
        selected = rng.choice(candidate_indices, size=num_random_samples, replace=False)
        mask_1d[selected] = 1.0

    # Expand 1D phase-encode mask to 2D k-space mask
    mask = np.repeat(mask_1d[:, None], width, axis=1)

    return mask.astype(np.float32)


def get_sampling_percentage(mask):
    """
    Return sampling percentage of a binary mask.
    """
    return 100.0 * float(mask.mean())
