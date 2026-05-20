"""
FRET Calibration Module
========================

Measures spectral bleedthrough correction factors from single-fluorophore
control cells:
    - alpha (α): CFP bleedthrough into the FRET channel
    - delta (δ): Direct YFP excitation by the CFP excitation laser
    - G-factor: Relates sensitized emission to donor quenching

Three-channel imaging setup:
    IDD: Donor excitation → Donor emission (e.g., 405nm → 450-490nm)
    IDA: Donor excitation → Acceptor emission (e.g., 405nm → 520-560nm)  [FRET channel]
    IAA: Acceptor excitation → Acceptor emission (e.g., 514nm → 520-560nm)

Control cells needed:
    - CFP-only cells → measure α
    - YFP-only cells → measure δ
    - Tandem construct or photobleaching → measure G
"""

import numpy as np
from scipy import stats as sp_stats
from skimage import filters
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime


class FRETCalibration:
    """
    Measure and store FRET bleedthrough correction factors.

    Parameters
    ----------
    output_dir : str or Path
        Directory for saving calibration results and QC plots.
    """

    def __init__(self, output_dir='./calibration'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.alpha = None      # CFP bleedthrough
        self.delta = None      # YFP direct excitation
        self.g_factor = None   # G-factor
        self.metadata = {}

    def _make_mask(self, image, method='otsu', min_intensity_percentile=10):
        """
        Create a foreground mask to exclude background pixels.

        Parameters
        ----------
        image : 2D array
            Image to threshold.
        method : str
            'otsu' or 'percentile'.
        min_intensity_percentile : float
            For 'percentile' method, exclude pixels below this percentile.

        Returns
        -------
        mask : 2D bool array
        """
        if method == 'otsu':
            try:
                thresh = filters.threshold_otsu(image)
                return image > thresh
            except ValueError:
                # Otsu fails on uniform images
                return image > np.percentile(image, min_intensity_percentile)
        else:
            return image > np.percentile(image, min_intensity_percentile)

    def measure_alpha(self, idd_cfponly, ida_cfponly, mask=None,
                       min_percentile=10, make_plot=True):
        """
        Measure CFP bleedthrough coefficient (α) from CFP-only control cells.

        In CFP-only cells, the FRET channel should contain ONLY bleedthrough
        from CFP emission (no actual FRET, no YFP). So: IDA = α × IDD.

        Parameters
        ----------
        idd_cfponly : 2D array or list of 2D arrays
            Donor channel image(s) from CFP-only cells.
        ida_cfponly : 2D array or list of 2D arrays
            FRET channel image(s) from CFP-only cells.
        mask : 2D bool array, optional
            Cell mask. If None, auto-generated from Otsu threshold on IDD.
        min_percentile : float
            Exclude dimmest pixels below this percentile.
        make_plot : bool
            Generate QC plot.

        Returns
        -------
        alpha : float
            CFP bleedthrough coefficient. Typical range: 0.02 - 0.05.
        """
        # Handle single image or list of images
        if isinstance(idd_cfponly, np.ndarray) and idd_cfponly.ndim == 2:
            idd_list = [idd_cfponly]
            ida_list = [ida_cfponly]
        else:
            idd_list = list(idd_cfponly)
            ida_list = list(ida_cfponly)

        all_idd = []
        all_ida = []

        for idd, ida in zip(idd_list, ida_list):
            idd = idd.astype(float)
            ida = ida.astype(float)

            if mask is None:
                m = self._make_mask(idd, min_intensity_percentile=min_percentile)
            else:
                m = mask

            idd_pixels = idd[m]
            ida_pixels = ida[m]

            # Further filter out very dim pixels
            bright_mask = idd_pixels > np.percentile(idd_pixels, min_percentile)
            all_idd.extend(idd_pixels[bright_mask])
            all_ida.extend(ida_pixels[bright_mask])

        all_idd = np.array(all_idd)
        all_ida = np.array(all_ida)

        # Robust linear regression (using median-based approach for outlier resistance)
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            all_idd, all_ida
        )

        self.alpha = slope
        self.metadata['alpha_r_squared'] = r_value ** 2
        self.metadata['alpha_n_pixels'] = len(all_idd)
        self.metadata['alpha_std_err'] = std_err

        print(f"α (CFP bleedthrough): {slope:.5f}")
        print(f"  R² = {r_value**2:.4f}  |  n = {len(all_idd)} pixels  |  SE = {std_err:.6f}")

        if r_value ** 2 < 0.90:
            print("  ⚠ WARNING: R² < 0.90 — check CFP-only control quality")

        if make_plot:
            self._plot_calibration(
                all_idd, all_ida, slope, intercept, r_value ** 2,
                xlabel='IDD (Donor channel)',
                ylabel='IDA (FRET channel)',
                title=f'α Calibration (CFP bleedthrough)\nα = {slope:.5f}, R² = {r_value**2:.4f}',
                filename='alpha_calibration.png'
            )

        return slope

    def measure_delta(self, ida_yfponly, iaa_yfponly, mask=None,
                       min_percentile=10, make_plot=True):
        """
        Measure YFP direct excitation coefficient (δ) from YFP-only controls.

        In YFP-only cells excited at the CFP wavelength (e.g. 405nm), any signal
        in the FRET channel comes from direct YFP excitation: IDA = δ × IAA.

        Parameters
        ----------
        ida_yfponly : 2D array or list of 2D arrays
            FRET channel from YFP-only cells (excited at donor wavelength).
        iaa_yfponly : 2D array or list of 2D arrays
            Acceptor channel from YFP-only cells (excited at acceptor wavelength).
        mask : 2D bool array, optional
            Cell mask.
        min_percentile : float
            Exclude dimmest pixels.
        make_plot : bool
            Generate QC plot.

        Returns
        -------
        delta : float
            YFP direct excitation coefficient. Typical range: 0.01 - 0.10.
        """
        if isinstance(ida_yfponly, np.ndarray) and ida_yfponly.ndim == 2:
            ida_list = [ida_yfponly]
            iaa_list = [iaa_yfponly]
        else:
            ida_list = list(ida_yfponly)
            iaa_list = list(iaa_yfponly)

        all_iaa = []
        all_ida = []

        for ida, iaa in zip(ida_list, iaa_list):
            ida = ida.astype(float)
            iaa = iaa.astype(float)

            if mask is None:
                m = self._make_mask(iaa, min_intensity_percentile=min_percentile)
            else:
                m = mask

            iaa_pixels = iaa[m]
            ida_pixels = ida[m]

            bright_mask = iaa_pixels > np.percentile(iaa_pixels, min_percentile)
            all_iaa.extend(iaa_pixels[bright_mask])
            all_ida.extend(ida_pixels[bright_mask])

        all_iaa = np.array(all_iaa)
        all_ida = np.array(all_ida)

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            all_iaa, all_ida
        )

        self.delta = slope
        self.metadata['delta_r_squared'] = r_value ** 2
        self.metadata['delta_n_pixels'] = len(all_iaa)
        self.metadata['delta_std_err'] = std_err

        print(f"δ (YFP direct excitation): {slope:.5f}")
        print(f"  R² = {r_value**2:.4f}  |  n = {len(all_iaa)} pixels  |  SE = {std_err:.6f}")

        if r_value ** 2 < 0.90:
            print("  ⚠ WARNING: R² < 0.90 — check YFP-only control quality")

        if make_plot:
            self._plot_calibration(
                all_iaa, all_ida, slope, intercept, r_value ** 2,
                xlabel='IAA (Acceptor channel)',
                ylabel='IDA (FRET channel)',
                title=f'δ Calibration (YFP direct excitation)\nδ = {slope:.5f}, R² = {r_value**2:.4f}',
                filename='delta_calibration.png'
            )

        return slope

    def set_g_factor(self, value, method='literature', notes=''):
        """
        Set the G-factor for FRET efficiency calculation.

        Parameters
        ----------
        value : float
            G-factor. Typical range for CFP-YFP: 0.5 - 0.8.
        method : str
            How it was determined: 'photobleaching', 'tandem', or 'literature'.
        notes : str
            Additional notes (e.g., microscope model, reference paper).
        """
        self.g_factor = value
        self.metadata['g_factor_method'] = method
        self.metadata['g_factor_notes'] = notes
        print(f"G-factor set: {value:.4f} (method: {method})")

    def measure_g_from_photobleaching(self, idd_pre, idd_post, ida_pre, iaa_pre):
        """
        Measure G-factor from acceptor photobleaching experiment.

        Bleach YFP in dual-expressing cells. The increase in CFP (donor dequenching)
        reveals the FRET efficiency, from which G can be calculated.

        Parameters
        ----------
        idd_pre : 2D array
            Donor channel BEFORE photobleaching.
        idd_post : 2D array
            Donor channel AFTER photobleaching YFP.
        ida_pre : 2D array
            FRET channel BEFORE photobleaching.
        iaa_pre : 2D array
            Acceptor channel BEFORE photobleaching.

        Returns
        -------
        g_factor : float
        """
        if self.alpha is None or self.delta is None:
            raise ValueError("Must calibrate alpha and delta before measuring G-factor")

        idd_pre = idd_pre.astype(float)
        idd_post = idd_post.astype(float)
        ida_pre = ida_pre.astype(float)
        iaa_pre = iaa_pre.astype(float)

        # Corrected FRET before bleaching
        fc = ida_pre - self.alpha * idd_pre - self.delta * iaa_pre

        # Donor recovery
        delta_idd = idd_post - idd_pre

        # Mask to cell region with positive signals
        mask = (fc > 0) & (delta_idd > 0) & (idd_pre > np.percentile(idd_pre, 20))

        # G = ΔI_DD / Fc
        g_values = delta_idd[mask] / fc[mask]

        # Use median for robustness
        g = np.median(g_values)

        self.g_factor = g
        self.metadata['g_factor_method'] = 'photobleaching'
        self.metadata['g_factor_median'] = float(g)
        self.metadata['g_factor_std'] = float(np.std(g_values))
        self.metadata['g_factor_n_pixels'] = int(mask.sum())

        print(f"G-factor (photobleaching): {g:.4f} ± {np.std(g_values):.4f}")
        print(f"  n = {mask.sum()} pixels")

        return g

    def is_complete(self):
        """Check if all calibration factors are set."""
        return all(v is not None for v in [self.alpha, self.delta, self.g_factor])

    def get_params(self):
        """Return calibration parameters as a dict."""
        return {
            'alpha': self.alpha,
            'delta': self.delta,
            'g_factor': self.g_factor
        }

    def save(self, filename='fret_calibration.json'):
        """Save calibration to JSON."""
        filepath = self.output_dir / filename

        data = {
            'alpha': self.alpha,
            'delta': self.delta,
            'g_factor': self.g_factor,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Calibration saved to {filepath}")

    def load(self, filepath):
        """Load calibration from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.alpha = data['alpha']
        self.delta = data['delta']
        self.g_factor = data['g_factor']
        self.metadata = data.get('metadata', {})

        print(f"Calibration loaded:")
        print(f"  α = {self.alpha}")
        print(f"  δ = {self.delta}")
        print(f"  G = {self.g_factor}")

    def _plot_calibration(self, x, y, slope, intercept, r_squared,
                           xlabel, ylabel, title, filename,
                           max_points=50000):
        """Generate calibration scatter plot with linear fit."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Subsample for plotting if too many points
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y

        # Scatter + fit line
        axes[0].scatter(x_plot, y_plot, alpha=0.1, s=1, c='steelblue')
        x_line = np.array([x.min(), x.max()])
        axes[0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
                     label=f'slope = {slope:.5f}\nR² = {r_squared:.4f}')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(title)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Residual histogram
        residuals = y - (slope * x + intercept)
        axes[1].hist(residuals, bins=100, edgecolor='none', color='steelblue', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Residuals (σ = {residuals.std():.2f})')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  QC plot saved: {self.output_dir / filename}")
