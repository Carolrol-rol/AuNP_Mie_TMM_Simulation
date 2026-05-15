"""Gold optical constants from Johnson & Christy (1972).

Johnson, P. B. & Christy, R. W. (1972). Optical constants of the noble metals.
Physical Review B, 6(12), 4370-4379.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .plot_style import set_journal_style, save_figure, FIGURE_SIZES

_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'johnson_christy_gold.csv')

# Module-level cached interpolators — built once on first import
_jc_df = None
_f_n   = None
_f_k   = None


def _ensure_loaded():
    """Load and cache J&C interpolators on first use."""
    global _jc_df, _f_n, _f_k
    if _f_n is None:
        _jc_df = pd.read_csv(_DATA_PATH).sort_values('wavelength_nm').reset_index(drop=True)
        _f_n = interp1d(_jc_df['wavelength_nm'], _jc_df['n'], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        _f_k = interp1d(_jc_df['wavelength_nm'], _jc_df['k'], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')


def load_jc_data():
    """Load the Johnson & Christy (1972) gold optical constants.

    Returns
    -------
    pandas.DataFrame
        Columns: wavelength_nm, n, k
    """
    _ensure_loaded()
    return _jc_df.copy()


def get_gold_nk(wavelength_nm_array):
    """Interpolate gold n and k at arbitrary wavelengths.

    Parameters
    ----------
    wavelength_nm_array : array-like
        Wavelengths in nm. Must lie in [400, 1000] nm.

    Returns
    -------
    n_array, k_array : numpy.ndarray
        Real refractive index and extinction coefficient.

    Raises
    ------
    ValueError
        If any wavelength is outside the physically relevant [400, 1000] nm range.
    """
    _ensure_loaded()
    wl = np.asarray(wavelength_nm_array, dtype=float)
    if np.any(wl < 400) or np.any(wl > 1000):
        raise ValueError(
            "Wavelengths must be in [400, 1000] nm (AuNP LSPR region). "
            f"Got range [{wl.min():.1f}, {wl.max():.1f}] nm."
        )
    return _f_n(wl), _f_k(wl)


def get_gold_permittivity(wavelength_nm_array):
    """Compute complex permittivity of gold.

    Parameters
    ----------
    wavelength_nm_array : array-like
        Wavelengths in nm (must be in [400, 1000] nm).

    Returns
    -------
    epsilon_Au : numpy.ndarray of complex
        Complex permittivity ε = (n + ik)^2.
    """
    n, k = get_gold_nk(wavelength_nm_array)
    return (n + 1j * k) ** 2


def plot_jc_data(save_path='figures/fig0_gold_optical_constants'):
    """Plot gold n and k from J&C (1972) over the AuNP LSPR window.

    Parameters
    ----------
    save_path : str
        Base path for output files (no extension).
    """
    set_journal_style()
    df = load_jc_data()

    mask = (df['wavelength_nm'] >= 400) & (df['wavelength_nm'] <= 1000)
    df_vis = df[mask]

    wl_fine = np.linspace(400, 1000, 600)
    n_fine, k_fine = get_gold_nk(wl_fine)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES['single_column_tall'],
                                    sharex=True)

    lspr_kw = dict(xmin=500, xmax=700, alpha=0.15, color='gold',
                   label='AuNP LSPR window')

    for ax, vals, label, color in [
        (ax1, n_fine, 'n (real part)', '#1f77b4'),
        (ax2, k_fine, 'k (extinction coeff.)', '#d62728'),
    ]:
        ax.plot(wl_fine, vals, color=color, lw=1.5)
        ax.scatter(df_vis['wavelength_nm'], df_vis[label.split()[0]],
                   s=18, color=color, zorder=5)
        ax.axvspan(**lspr_kw)
        ax.set_ylabel(label, fontsize=10)

    ax1.text(600, ax1.get_ylim()[1] * 0.85, 'AuNP LSPR window',
             ha='center', va='top', fontsize=8, color='#8B6914')
    ax2.text(600, ax2.get_ylim()[1] * 0.15, 'AuNP LSPR window',
             ha='center', va='bottom', fontsize=8, color='#8B6914')

    ax2.set_xlabel('Wavelength (nm)', fontsize=10)
    ax2.set_xlim(400, 1000)
    ax1.set_title('Gold optical constants (Johnson & Christy 1972)', fontsize=11)

    save_figure(fig, save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_jc_data()
