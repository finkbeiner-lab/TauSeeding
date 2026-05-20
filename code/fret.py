"""
Pixel-Level FRET Computation Module
=====================================

Computes corrected FRET (Fc) and FRET efficiency (E) at every pixel
using the three-channel sensitized emission method.

The raw FRET channel (IDA) contains three components:
    IDA = True_FRET + α×IDD + δ×IAA

Where:
    α×IDD = CFP spectral bleedthrough into the FRET channel
    δ×IAA = Direct YFP excitation by the donor laser

Corrected FRET:
    Fc = IDA - α×IDD - δ×IAA

FRET Efficiency:
    E = Fc / (Fc + G×IDD)

This module operates pixel-by-pixel, preserving spatial information
about punctate Tau aggregates.
"""

import numpy as np
from scipy import ndimage


def subtract_background(image, method='percentile', percentile=5,
                         rolling_ball_radius=None):
    """
    Subtract background from a single channel image.

    Parameters
    ----------
    image : 2D array
        Raw intensity image.
    method : str
        'percentile' - subtract a fixed percentile value (fast, good for
                        uniform backgrounds)
        'rolling_ball' - rolling ball background estimation (handles
                          uneven illumination)
        'corner' - estimate from image corners (for images with cells
                    not touching edges)
    percentile : float
        For 'percentile' method. Default 5th percentile.
    rolling_ball_radius : int
        For 'rolling_ball' method. Radius in pixels.

    Returns
    -------
    bg_subtracted : 2D array
        Background-subtracted image, clipped to >= 0.
    bg_value : float or 2D array
        The estimated background.
    """
    img = image.astype(float)

    if method == 'percentile':
        bg = np.percentile(img, percentile)
        result = img - bg

    elif method == 'rolling_ball':
        if rolling_ball_radius is None:
            rolling_ball_radius = 50
        # Approximate rolling ball with large-kernel minimum + gaussian
        bg = ndimage.minimum_filter(img, size=rolling_ball_radius)
        bg = ndimage.gaussian_filter(bg, sigma=rolling_ball_radius / 2)
        result = img - bg

    elif method == 'corner':
        h, w = img.shape
        corner_size = max(10, min(h, w) // 20)
        corners = np.concatenate([
            img[:corner_size, :corner_size].ravel(),
            img[:corner_size, -corner_size:].ravel(),
            img[-corner_size:, :corner_size].ravel(),
            img[-corner_size:, -corner_size:].ravel()
        ])
        bg = np.median(corners)
        result = img - bg

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.maximum(result, 0), bg


def compute_corrected_fret(idd, ida, iaa, alpha, delta,
                            bg_method='percentile', bg_percentile=5):
    """
    Compute corrected FRET image (Fc) at every pixel.

    Fc = IDA - α×IDD - δ×IAA

    Parameters
    ----------
    idd : 2D array
        Donor channel (CFP excitation → CFP emission).
    ida : 2D array
        FRET channel (CFP excitation → YFP emission).
    iaa : 2D array
        Acceptor channel (YFP excitation → YFP emission).
    alpha : float
        CFP bleedthrough coefficient.
    delta : float
        YFP direct excitation coefficient.
    bg_method : str
        Background subtraction method ('percentile', 'rolling_ball', 'corner').
    bg_percentile : float
        Percentile for background estimation.

    Returns
    -------
    fc : 2D array
        Corrected FRET image. Negative values clipped to 0.
    bg_info : dict
        Background values for each channel (for QC).
    """
    # Background subtract all channels
    idd_bg, bg_idd = subtract_background(idd.astype(float), bg_method, bg_percentile)
    ida_bg, bg_ida = subtract_background(ida.astype(float), bg_method, bg_percentile)
    iaa_bg, bg_iaa = subtract_background(iaa.astype(float), bg_method, bg_percentile)

    # Corrected FRET
    fc = ida_bg - (alpha * idd_bg) - (delta * iaa_bg)

    # Clip negative to zero (no negative FRET)
    fc = np.maximum(fc, 0)

    bg_info = {
        'bg_idd': bg_idd,
        'bg_ida': bg_ida,
        'bg_iaa': bg_iaa,
        'idd_subtracted': idd_bg,
        'ida_subtracted': ida_bg,
        'iaa_subtracted': iaa_bg,
    }

    return fc, bg_info


def compute_fret_efficiency(fc, idd, g_factor, min_signal=0):
    """
    Compute FRET efficiency map at every pixel.

    E = Fc / (Fc + G × IDD)

    Parameters
    ----------
    fc : 2D array
        Corrected FRET image (from compute_corrected_fret).
    idd : 2D array
        Background-subtracted donor image.
    g_factor : float
        G-factor relating sensitized emission to donor quenching.
    min_signal : float
        Minimum total signal (Fc + G*IDD) to compute efficiency.
        Pixels below this threshold are set to 0 (avoids noise
        amplification in dim regions).

    Returns
    -------
    efficiency : 2D array
        FRET efficiency map. Range [0, 1] where signal is present,
        0 elsewhere.
    signal_mask : 2D bool array
        True where signal was sufficient to compute efficiency.
    """
    denominator = fc + g_factor * idd

    # Create mask for reliable pixels
    if min_signal > 0:
        signal_mask = denominator > min_signal
    else:
        signal_mask = denominator > 0

    # Compute efficiency where denominator is valid
    efficiency = np.zeros_like(fc)
    efficiency[signal_mask] = fc[signal_mask] / denominator[signal_mask]

    # Clip to [0, 1] — values outside indicate calibration issues
    n_over_1 = np.sum(efficiency > 1.0)
    if n_over_1 > 0:
        pct = 100 * n_over_1 / signal_mask.sum() if signal_mask.sum() > 0 else 0
        if pct > 1:
            print(f"  ⚠ {pct:.1f}% of pixels have E > 1.0 — check calibration factors")

    efficiency = np.clip(efficiency, 0, 1)

    return efficiency, signal_mask


def compute_nfret(fc, idd, iaa):
    """
    Compute normalized FRET (NFRET), which accounts for
    donor and acceptor expression levels.

    NFRET = Fc / sqrt(IDD × IAA)

    This is useful when expression levels vary between cells,
    as it normalizes out concentration effects.

    Parameters
    ----------
    fc : 2D array
        Corrected FRET image.
    idd : 2D array
        Background-subtracted donor image.
    iaa : 2D array
        Background-subtracted acceptor image.

    Returns
    -------
    nfret : 2D array
        Normalized FRET image.
    """
    product = idd * iaa
    valid = product > 0

    nfret = np.zeros_like(fc)
    nfret[valid] = fc[valid] / np.sqrt(product[valid])

    return nfret


def compute_fret_index(fc, idd):
    """
    Compute simple FRET index (ratio).

    FRET_index = Fc / IDD

    Simpler than efficiency (no G-factor needed), useful for
    relative comparisons when G is unknown.

    Parameters
    ----------
    fc : 2D array
        Corrected FRET image.
    idd : 2D array
        Background-subtracted donor image.

    Returns
    -------
    fret_index : 2D array
    """
    valid = idd > 0
    fret_index = np.zeros_like(fc)
    fret_index[valid] = fc[valid] / idd[valid]
    return fret_index


def process_fov(idd, ida, iaa, alpha, delta, g_factor,
                 bg_method='percentile', min_signal=50):
    """
    Complete FRET processing for one field of view.

    Convenience function that runs background subtraction,
    corrected FRET, FRET efficiency, and NFRET in one call.

    Parameters
    ----------
    idd, ida, iaa : 2D arrays
        Three-channel images for one FOV.
    alpha, delta : float
        Bleedthrough correction factors.
    g_factor : float
        G-factor for efficiency calculation.
    bg_method : str
        Background subtraction method.
    min_signal : float
        Minimum signal threshold for efficiency map.

    Returns
    -------
    results : dict
        Contains:
        - 'fc': corrected FRET image
        - 'efficiency': FRET efficiency map
        - 'nfret': normalized FRET
        - 'fret_index': simple FRET ratio
        - 'signal_mask': where efficiency is reliable
        - 'idd_bg': background-subtracted donor
        - 'iaa_bg': background-subtracted acceptor
    """
    # Corrected FRET
    fc, bg_info = compute_corrected_fret(
        idd, ida, iaa, alpha, delta,
        bg_method=bg_method
    )

    idd_bg = bg_info['idd_subtracted']
    iaa_bg = bg_info['iaa_subtracted']

    # FRET efficiency
    efficiency, signal_mask = compute_fret_efficiency(
        fc, idd_bg, g_factor, min_signal=min_signal
    )

    # Normalized FRET
    nfret = compute_nfret(fc, idd_bg, iaa_bg)

    # Simple FRET index
    fret_index = compute_fret_index(fc, idd_bg)

    return {
        'fc': fc,
        'efficiency': efficiency,
        'nfret': nfret,
        'fret_index': fret_index,
        'signal_mask': signal_mask,
        'idd_bg': idd_bg,
        'ida_bg': bg_info['ida_subtracted'],
        'iaa_bg': iaa_bg,
    }
