"""Mie theory module for AuNP optical properties using PyMieScatt.

Implements exact Mie solutions for spherical gold nanoparticles in a
homogeneous surrounding medium.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import PyMieScatt as ps
    _HAS_PYMIE = True
except ImportError:
    _HAS_PYMIE = False
    warnings.warn(
        "PyMieScatt not found. Mie calculations will use the built-in "
        "Bohren & Huffman series implementation.", ImportWarning
    )

from .gold_dielectric import get_gold_nk
from .plot_style import set_journal_style, save_figure, FIGURE_SIZES

# Physical constants
_WL_MIN  = 400.0   # nm  — lower bound for AuNP LSPR
_WL_MAX  = 900.0   # nm  — upper bound
_WL_STEP = 1.0     # nm — plotting resolution
_WL_STEP_SENS = 2.0  # nm — coarser grid for sensitivity sweeps (faster)


# ---------------------------------------------------------------------------
# Internal fallback: Bohren & Huffman Mie series (used when PyMieScatt absent)
# ---------------------------------------------------------------------------

def _mie_coeffs(m, x):
    """Compute Mie scattering coefficients a_n, b_n up to n_max terms.

    Parameters
    ----------
    m : complex
        Relative refractive index (particle / medium).
    x : float
        Size parameter = pi * d / lambda.

    Returns
    -------
    Qext, Qsca : float
    """
    from scipy.special import spherical_jn, spherical_yn
    nmax = int(np.round(2 + x + 4 * x ** (1 / 3))) + 2
    n = np.arange(1, nmax + 1)
    mx = m * x

    def psi(n, z):
        return z * spherical_jn(n, z)

    def xi(n, z):
        return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))

    def dpsi(n, z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    def dxi(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z) +
                z * (spherical_jn(n, z, derivative=True) +
                     1j * spherical_yn(n, z, derivative=True)))

    a_n = (m * psi(n, mx) * dpsi(n, x) - psi(n, x) * dpsi(n, mx)) / \
          (m * psi(n, mx) * dxi(n, x) - xi(n, x) * dpsi(n, mx))
    b_n = (psi(n, mx) * dpsi(n, x) - m * psi(n, x) * dpsi(n, mx)) / \
          (psi(n, mx) * dxi(n, x) - m * xi(n, x) * dpsi(n, mx))

    Qext = (2 / x ** 2) * np.sum((2 * n + 1) * np.real(a_n + b_n))
    Qsca = (2 / x ** 2) * np.sum((2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))
    return float(Qext), float(Qsca)


def _compute_mie_bh(diameter_nm, n_medium, wavelength_nm):
    """Bohren & Huffman Mie fallback for a single wavelength."""
    n_au, k_au = get_gold_nk(np.array([wavelength_nm]))
    m_particle = complex(n_au[0], k_au[0])
    m_rel = m_particle / n_medium
    x = np.pi * diameter_nm / wavelength_nm
    Qext, Qsca = _mie_coeffs(m_rel, x)
    Qabs = Qext - Qsca
    return Qext, Qsca, Qabs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_extinction_spectrum(diameter_nm, n_medium, wavelengths_nm=None):
    """Compute Mie extinction spectrum for a gold nanosphere.

    Qext = Qsca + Qabs (extinction = scattering + absorption).

    Parameters
    ----------
    diameter_nm : float
        AuNP diameter in nm.
    n_medium : float
        Surrounding medium refractive index.
    wavelengths_nm : array-like, optional
        Wavelengths in nm. Defaults to 400–900 nm, step 0.5 nm.

    Returns
    -------
    dict
        Keys: 'wavelength', 'Qext', 'Qsca', 'Qabs'
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.arange(_WL_MIN, _WL_MAX + _WL_STEP, _WL_STEP)
    wl = np.asarray(wavelengths_nm, dtype=float)

    Qext_arr = np.zeros(len(wl))
    Qsca_arr = np.zeros(len(wl))
    Qabs_arr = np.zeros(len(wl))

    # Precompute all n,k values in a single vectorised call (avoids repeated CSV reads)
    try:
        n_all, k_all = get_gold_nk(wl)
    except ValueError:
        n_all, k_all = np.full(len(wl), np.nan), np.full(len(wl), np.nan)

    for i, lam in enumerate(wl):
        try:
            m_particle = complex(n_all[i], k_all[i])
            if np.isnan(m_particle.real):
                raise ValueError("J&C out of range")

            if _HAS_PYMIE:
                # Pass absolute m; PyMieScatt.MieQ divides by nMedium internally
                result = ps.MieQ(m_particle, lam, diameter_nm,
                                 nMedium=n_medium, asCrossSection=False)
                Qext_arr[i] = result[0]
                Qsca_arr[i] = result[1]
                Qabs_arr[i] = result[2]
            else:
                Qext_arr[i], Qsca_arr[i], Qabs_arr[i] = _compute_mie_bh(
                    diameter_nm, n_medium, lam)
        except (ValueError, RuntimeError):
            Qext_arr[i] = Qsca_arr[i] = Qabs_arr[i] = np.nan

    return {'wavelength': wl, 'Qext': Qext_arr, 'Qsca': Qsca_arr, 'Qabs': Qabs_arr}


def find_lspr_peak(wavelengths_nm, Qext_array):
    """Find LSPR peak wavelength, amplitude, and FWHM.

    Parameters
    ----------
    wavelengths_nm : array-like
    Qext_array : array-like

    Returns
    -------
    tuple
        (peak_wavelength_nm, peak_Qext, FWHM_nm)
    """
    wl = np.asarray(wavelengths_nm)
    Q  = np.asarray(Qext_array)
    valid = np.isfinite(Q)
    wl, Q = wl[valid], Q[valid]

    peaks, props = find_peaks(Q, prominence=0.05)
    if len(peaks) == 0:
        idx = np.argmax(Q)
        peaks = np.array([idx])

    # Take peak with highest Qext in the 450–750 nm LSPR window
    lspr_mask = (wl[peaks] >= 450) & (wl[peaks] <= 750)
    if lspr_mask.any():
        peak_idx = peaks[lspr_mask][np.argmax(Q[peaks[lspr_mask]])]
    else:
        peak_idx = peaks[np.argmax(Q[peaks])]

    # Sub-pixel peak refinement: parabolic interpolation through 3 points around max
    if 0 < peak_idx < len(wl) - 1:
        y0, y1, y2 = Q[peak_idx - 1], Q[peak_idx], Q[peak_idx + 1]
        x0, x1, x2 = wl[peak_idx - 1], wl[peak_idx], wl[peak_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-15:
            offset = 0.5 * (y0 - y2) / denom
            peak_wl = x1 + offset * (x2 - x1)
            peak_Q  = y1 - 0.25 * (y0 - y2) * offset
        else:
            peak_wl, peak_Q = wl[peak_idx], Q[peak_idx]
    else:
        peak_wl, peak_Q = wl[peak_idx], Q[peak_idx]

    half_max = peak_Q / 2.0

    # FWHM via linear interpolation
    left_idx  = np.searchsorted(Q[:peak_idx][::-1], half_max)
    right_idx = np.searchsorted(-Q[peak_idx:], -half_max)

    try:
        wl_left  = wl[peak_idx - left_idx]
        wl_right = wl[peak_idx + right_idx]
        FWHM = wl_right - wl_left
    except IndexError:
        FWHM = np.nan

    return float(peak_wl), float(peak_Q), float(FWHM)


def compute_bulk_ri_sensitivity(diameter_nm, n_med_range=None):
    """Compute bulk refractive index sensitivity for a given AuNP diameter.

    Fits Δλ = S × Δn via linear regression across a range of medium RIs.

    Parameters
    ----------
    diameter_nm : float
    n_med_range : list of float, optional
        Medium RI values to use. Defaults to [1.333, 1.343, 1.353, 1.363, 1.373].

    Returns
    -------
    dict
        Keys: 'sensitivity_nm_per_RIU', 'R_squared', 'peak_wavelengths', 'n_values'
    """
    if n_med_range is None:
        n_med_range = [1.333, 1.343, 1.353, 1.363, 1.373]
    n_vals = np.array(n_med_range)
    peak_wls = []
    # Coarser wavelength grid is sufficient for peak detection
    wl_sens = np.arange(_WL_MIN, _WL_MAX + _WL_STEP_SENS, _WL_STEP_SENS)

    for n_med in n_vals:
        spec = compute_extinction_spectrum(diameter_nm, n_med, wl_sens)
        peak_wl, _, _ = find_lspr_peak(spec['wavelength'], spec['Qext'])
        peak_wls.append(peak_wl)

    peak_wls = np.array(peak_wls)
    slope, intercept, r, *_ = linregress(n_vals, peak_wls)
    return {
        'sensitivity_nm_per_RIU': float(slope),
        'R_squared':              float(r ** 2),
        'peak_wavelengths':       peak_wls,
        'n_values':               n_vals,
    }


def sensitivity_vs_diameter(d_array=None, n_med=1.333):
    """Compute LSPR sensitivity and FOM for a range of AuNP diameters.

    Parameters
    ----------
    d_array : array-like, optional
        Diameters in nm. Defaults to np.arange(20, 81, 5).
    n_med : float
        Reference medium RI for peak wavelength and FWHM.

    Returns
    -------
    dict
        Keys: 'diameters', 'sensitivity', 'FWHM', 'FOM'
    """
    if d_array is None:
        d_array = np.arange(20, 81, 5)
    d_array = np.asarray(d_array)

    wl_sens = np.arange(_WL_MIN, _WL_MAX + _WL_STEP_SENS, _WL_STEP_SENS)
    sensitivities, FWHMs, FOMs = [], [], []
    for d in d_array:
        sens_data = compute_bulk_ri_sensitivity(d)
        S = sens_data['sensitivity_nm_per_RIU']
        spec = compute_extinction_spectrum(d, n_med, wl_sens)
        _, _, FWHM = find_lspr_peak(spec['wavelength'], spec['Qext'])
        FOM = compute_fom(S, FWHM) if (FWHM and np.isfinite(FWHM) and FWHM > 0) else np.nan
        sensitivities.append(S)
        FWHMs.append(FWHM)
        FOMs.append(FOM)

    return {
        'diameters':   d_array,
        'sensitivity': np.array(sensitivities),
        'FWHM':        np.array(FWHMs),
        'FOM':         np.array(FOMs),
    }


def compute_fom(sensitivity_nm_per_RIU, FWHM_nm):
    """Compute Figure of Merit = sensitivity / FWHM.

    Higher FOM indicates sharper, more sensitive resonance.

    Parameters
    ----------
    sensitivity_nm_per_RIU : float
    FWHM_nm : float

    Returns
    -------
    float
    """
    return sensitivity_nm_per_RIU / FWHM_nm


def plot_extinction_spectra(save_path='figures/fig1_extinction_spectra'):
    """Plot Qext spectra for d = 20–80 nm AuNPs in water.

    Parameters
    ----------
    save_path : str
        Base path without extension.
    """
    set_journal_style()
    diameters = [20, 30, 40, 50, 60, 70, 80]
    wl = np.arange(400, 801, 0.5)
    cmap = cm.plasma
    norm = plt.Normalize(vmin=min(diameters), vmax=max(diameters))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])
    for d in diameters:
        spec = compute_extinction_spectrum(d, 1.333, wl)
        color = cmap(norm(d))
        ax.plot(spec['wavelength'], spec['Qext'], color=color, lw=1.5)
        peak_wl, peak_Q, _ = find_lspr_peak(spec['wavelength'], spec['Qext'])
        ax.axvline(peak_wl, color=color, ls='--', lw=0.8, alpha=0.6)
        ax.annotate(f'{peak_wl:.0f}', xy=(peak_wl, peak_Q),
                    xytext=(peak_wl + 8, peak_Q * 0.95),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.7))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('AuNP diameter (nm)', fontsize=9)

    ax.annotate('', xy=(700, ax.get_ylim()[1] * 0.6),
                xytext=(550, ax.get_ylim()[1] * 0.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(625, ax.get_ylim()[1] * 0.65, 'Redshift', ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Extinction efficiency $Q_{ext}$')
    ax.set_xlim(400, 800)
    ax.set_title('AuNP Mie extinction spectra (n = 1.333)')

    save_figure(fig, save_path)
    plt.close(fig)


def plot_sensitivity_vs_diameter(save_path='figures/fig2_sensitivity_diameter'):
    """Plot sensitivity (nm/RIU) and FOM vs AuNP diameter.

    Parameters
    ----------
    save_path : str
        Base path without extension.
    """
    set_journal_style()
    data = sensitivity_vs_diameter(np.arange(20, 81, 5))
    d = data['diameters']
    S = data['sensitivity']
    FOM = data['FOM']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES['single_column_tall'],
                                    sharex=True)

    ax1.plot(d, S, 'o-', color='#1f77b4', lw=1.5)
    ax1.axhline(100, ls=':', color='gray', lw=1)
    ax1.axhline(200, ls=':', color='gray', lw=1)
    ax1.text(d[-1] + 1, 100, 'Typical experimental\nrange', fontsize=7,
             va='center', color='gray')
    ax1.set_ylabel('Sensitivity (nm/RIU)')

    fom_valid = np.where(np.isfinite(FOM))
    ax2.plot(d[fom_valid], FOM[fom_valid], 's-', color='#d62728', lw=1.5)

    if np.any(np.isfinite(FOM)):
        opt_idx = np.nanargmax(FOM)
        ax2.plot(d[opt_idx], FOM[opt_idx], 'r*', ms=14, zorder=5,
                 label=f'Optimal: d={d[opt_idx]:.0f} nm, FOM={FOM[opt_idx]:.2f}')
        ax2.legend(fontsize=8)

    ax2.set_xlabel('AuNP diameter (nm)')
    ax2.set_ylabel('Figure of Merit (FOM)')
    ax1.set_title('LSPR sensitivity vs AuNP diameter')

    save_figure(fig, save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_extinction_spectra()
    plot_sensitivity_vs_diameter()
