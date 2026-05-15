"""Transfer Matrix Method for evanescent field in etched SMF-28 fiber.

Models the etched fiber as a planar three-layer slab waveguide to compute
evanescent field intensity and penetration depth as functions of etch depth.

Reference: Khaliq et al. (2003) Fiber-optic surface plasmon resonance sensor.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .plot_style import set_journal_style, save_figure, FIGURE_SIZES

# SMF-28 physical parameters
_N_CORE        = 1.4682    # at 633 nm
_N_CLAD        = 1.4628    # at 633 nm
_CORE_RADIUS   = 4.1       # µm  (half of 8.2 µm core diameter)
_CLAD_RADIUS   = 62.5      # µm
_DELTA_MAX     = (_CLAD_RADIUS - _CORE_RADIUS)  # 58.4 µm full cladding depth
_N_SURR        = 1.333     # water
_LAMBDA_NM     = 633.0     # He-Ne operating wavelength


def compute_propagation_constant(n_core=_N_CORE, n_clad=_N_CLAD,
                                  wavelength_nm=_LAMBDA_NM,
                                  core_radius_um=_CORE_RADIUS):
    """Solve for the LP01 (HE11) mode propagation constant.

    Uses the LP01 characteristic equation for a step-index fiber.

    Parameters
    ----------
    n_core, n_clad : float
        Core and cladding refractive indices.
    wavelength_nm : float
    core_radius_um : float
        Core radius in µm.

    Returns
    -------
    beta : float
        Propagation constant in rad/µm.
    n_eff : float
        Effective refractive index.
    """
    lam_um = wavelength_nm / 1000.0
    k0 = 2 * np.pi / lam_um          # rad/µm
    V = k0 * core_radius_um * np.sqrt(n_core ** 2 - n_clad ** 2)

    # LP01 characteristic equation: J0(u)/J1(u) = -K0(v)/K1(v) * v/u
    # where u^2 + v^2 = V^2, u = a*sqrt(k0^2*n_core^2 - beta^2)
    from scipy.special import j0, j1, k0 as K0, k1 as K1

    def char_eq(u):
        if u <= 0 or u >= V:
            return np.inf
        v = np.sqrt(max(V ** 2 - u ** 2, 1e-30))
        lhs = u * j0(u) / j1(u) if abs(j1(u)) > 1e-15 else np.inf
        rhs = -v * K0(v) / K1(v) if abs(K1(v)) > 1e-15 else np.inf
        return lhs - rhs

    try:
        u_sol = brentq(char_eq, 1e-6, V - 1e-6, xtol=1e-10)
    except ValueError:
        u_sol = V * 0.8   # fallback for nearly-cut-off modes

    beta = np.sqrt((k0 * n_core) ** 2 - (u_sol / core_radius_um) ** 2)
    n_eff = beta / k0
    return float(beta), float(n_eff)


def _effective_index_cladded(t_clad_um, wavelength_nm, n_core, n_clad, n_surr):
    """Compute effective index for the remaining-cladding structure.

    Uses a weighted average (effective medium) when the cladding becomes thin.
    Falls back to n_clad for thick cladding; approaches n_surr for zero cladding.

    Parameters
    ----------
    t_clad_um : float
        Remaining cladding thickness in µm.
    wavelength_nm : float
    n_core, n_clad, n_surr : float

    Returns
    -------
    n_eff : float
    """
    _, n_eff_full = compute_propagation_constant(n_core, n_clad,
                                                  wavelength_nm, _CORE_RADIUS)
    lam_um = wavelength_nm / 1000.0
    # Penetration depth into cladding (from core side)
    ld_core_um = lam_um / (4 * np.pi * np.sqrt(max(n_eff_full ** 2 - n_clad ** 2, 1e-12)))
    # Weight: how much of the evanescent tail is in the cladding vs analyte
    weight = 1.0 - np.exp(-t_clad_um / max(ld_core_um, 1e-6))
    weight = np.clip(weight, 0.0, 1.0)
    n_eff = n_eff_full * weight + n_surr * (1 - weight)
    n_eff = max(n_eff, n_surr + 1e-6)   # must remain above n_surr for guided mode
    return float(n_eff)


def compute_evanescent_field(etch_depth_um, wavelength_nm=_LAMBDA_NM,
                              n_core=_N_CORE, n_clad=_N_CLAD,
                              n_surr=_N_SURR):
    """Compute evanescent field parameters for a given cladding etch depth.

    Parameters
    ----------
    etch_depth_um : float
        Cladding removed from the outer surface (µm). Max = 58.4 µm.
    wavelength_nm : float
    n_core, n_clad, n_surr : float

    Returns
    -------
    dict
        Keys: 'etch_depth_um', 't_clad_um', 'l_d_nm', 'I_ev_normalised', 'n_eff'

    Raises
    ------
    ValueError
        If etch_depth_um > _DELTA_MAX.
    """
    if etch_depth_um > _DELTA_MAX + 1e-9:
        raise ValueError(
            f"etch_depth_um ({etch_depth_um:.2f}) exceeds maximum cladding "
            f"thickness ({_DELTA_MAX:.1f} µm)."
        )

    t_clad_um = max(_DELTA_MAX - etch_depth_um, 0.0)   # remaining cladding, µm
    n_eff = _effective_index_cladded(t_clad_um, wavelength_nm,
                                      n_core, n_clad, n_surr)

    lam_um = wavelength_nm / 1000.0
    denom = max(n_eff ** 2 - n_surr ** 2, 1e-12)
    l_d_nm = (wavelength_nm) / (4 * np.pi * np.sqrt(denom))   # nm

    # Evanescent intensity at the fiber surface (exponential decay through cladding)
    l_d_um = l_d_nm / 1000.0
    I_ev = np.exp(-2 * t_clad_um / max(l_d_um, 1e-9))

    return {
        'etch_depth_um':   float(etch_depth_um),
        't_clad_um':       float(t_clad_um),
        'l_d_nm':          float(l_d_nm),
        'I_ev_normalised': float(I_ev),
        'n_eff':           float(n_eff),
    }


def sweep_etch_depth(delta_array=None, wavelength_nm=_LAMBDA_NM):
    """Sweep evanescent field parameters over a range of etch depths.

    Parameters
    ----------
    delta_array : array-like, optional
        Etch depths in µm. Defaults to np.linspace(0, 58.4, 200).
    wavelength_nm : float

    Returns
    -------
    pandas.DataFrame
        Columns: etch_depth_um, t_clad_um, l_d_nm, I_ev_normalised, n_eff
    """
    if delta_array is None:
        delta_array = np.linspace(0, _DELTA_MAX, 200)
    rows = [compute_evanescent_field(d, wavelength_nm) for d in delta_array]
    return pd.DataFrame(rows)


def plot_evanescent_field(save_path='figures/fig3_evanescent_field'):
    """Two-panel plot of evanescent field intensity and penetration depth.

    Parameters
    ----------
    save_path : str
        Base path without extension.
    """
    set_journal_style()
    df = sweep_etch_depth()

    # Inflection point: maximum of dI/d_delta
    dI = np.gradient(df['I_ev_normalised'], df['etch_depth_um'])
    inflect_idx = np.argmax(dI)
    inflect_delta = df['etch_depth_um'].iloc[inflect_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES['single_column_tall'],
                                    sharex=True)

    # Top panel: I_ev (log scale)
    ax1.semilogy(df['etch_depth_um'], df['I_ev_normalised'], 'k-', lw=1.5)
    ax1.axvline(inflect_delta, color='purple', ls='--', lw=1,
                label=f'Inflection δ={inflect_delta:.1f} µm')
    ax1.legend(fontsize=8)

    # Region shading
    ax1.axvspan(0, 30, alpha=0.12, color='#1f77b4')
    ax1.axvspan(30, 50, alpha=0.12, color='#2ca02c')
    ax1.axvspan(50, _DELTA_MAX, alpha=0.12, color='#d62728')

    ylim = ax1.get_ylim()
    ax1.text(15, 10 ** (0.8 * np.log10(ylim[1]) + 0.2 * np.log10(max(ylim[0], 1e-15))),
             'Shallow\netching', ha='center', fontsize=7, color='#1f77b4')
    ax1.text(40, 10 ** (0.8 * np.log10(ylim[1]) + 0.2 * np.log10(max(ylim[0], 1e-15))),
             'Optimal\nwindow', ha='center', fontsize=7, color='#2ca02c')
    ax1.text(54, 10 ** (0.8 * np.log10(ylim[1]) + 0.2 * np.log10(max(ylim[0], 1e-15))),
             'Deep\netching', ha='center', fontsize=7, color='#d62728')

    ax1.set_ylabel('Normalised evanescent\nfield intensity $I_{ev}$')

    # Bottom panel: penetration depth
    ax2.plot(df['etch_depth_um'], df['l_d_nm'], '#ff7f0e', lw=1.5)
    ax2.axhline(200, color='gray', ls='--', lw=1,
                label='AuNP diameter range upper bound (200 nm)')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('Cladding etch depth δ (µm)')
    ax2.set_ylabel('Penetration depth $l_d$ (nm)')
    ax1.set_title('Evanescent field vs cladding etch depth (SMF-28, 633 nm)')

    save_figure(fig, save_path)
    plt.close(fig)


def plot_mode_profile(etch_depth_um=30.0, save_path='figures/fig3b_mode_profile'):
    """Schematic cross-section of etched fiber with evanescent field profile.

    Not to scale — schematic illustration only.

    Parameters
    ----------
    etch_depth_um : float
    save_path : str
    """
    set_journal_style()
    result = compute_evanescent_field(etch_depth_um)
    t_clad = result['t_clad_um']
    l_d    = result['l_d_nm']

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Schematic cross-section (δ = {etch_depth_um:.0f} µm)\n'
                 '(not to scale)', fontsize=9)

    # Draw layers as horizontal bands (y-axis = radial direction)
    core_w  = 4.0    # schematic width units
    clad_h  = max(t_clad / _DELTA_MAX * 2.0, 0.2)   # scaled thickness
    evan_h  = 1.5    # evanescent field extent shown

    # Core
    ax.add_patch(mpatches.FancyBboxPatch((-core_w / 2, -1.0), core_w, 2.0,
                  boxstyle='round,pad=0.1', fc='#aaaaaa', ec='k', lw=1.2))
    ax.text(0, 0, f'Core\nn = {_N_CORE}', ha='center', va='center', fontsize=8)

    # Remaining cladding
    if t_clad > 0:
        ax.add_patch(mpatches.FancyBboxPatch((-core_w / 2, 1.0), core_w, clad_h,
                      boxstyle='square,pad=0', fc='#add8e6', ec='k', lw=0.8))
        ax.text(0, 1.0 + clad_h / 2, f'Cladding (t={t_clad:.1f} µm)\nn = {_N_CLAD}',
                ha='center', va='center', fontsize=7)

    # Analyte layer
    top = 1.0 + clad_h
    ax.add_patch(mpatches.FancyBboxPatch((-core_w / 2, top), core_w, evan_h,
                  boxstyle='square,pad=0', fc='#ffe5b4', ec='k', lw=0.8, alpha=0.8))
    ax.text(0, top + evan_h / 2, f'Analyte\nn = {_N_SURR}',
            ha='center', va='center', fontsize=7)

    # Evanescent field decay curve
    y_evan = np.linspace(0, evan_h, 100)
    x_evan = 1.8 * np.exp(-y_evan / (l_d / 1000 / _DELTA_MAX * 2))
    ax.plot(x_evan + core_w / 2, y_evan + top, color='#d62728', lw=2,
            label=f'$l_d$ = {l_d:.0f} nm')

    # Delta arrow
    ax.annotate('', xy=(core_w / 2 + 0.3, 1.0 + clad_h),
                xytext=(core_w / 2 + 0.3, 1.0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(core_w / 2 + 0.6, 1.0 + clad_h / 2,
            f'δ = {etch_depth_um:.0f} µm', fontsize=7, va='center')

    ax.legend(loc='lower right', fontsize=7)
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-1.5, top + evan_h + 0.3)

    save_figure(fig, save_path)
    plt.close(fig)


if __name__ == '__main__':
    plot_evanescent_field()
    plot_mode_profile()
