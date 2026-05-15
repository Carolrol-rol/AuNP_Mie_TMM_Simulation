"""Parametric composite sensitivity figure.

Combines Mie theory (S_LSPR) and TMM evanescent field (I_ev) to generate
the central figure of the paper: S_total = S_LSPR(d) × I_ev(t_clad).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from .mie_theory import sensitivity_vs_diameter
from .transfer_matrix import sweep_etch_depth, compute_evanescent_field, _DELTA_MAX
from .plot_style import set_journal_style, save_figure, FIGURE_SIZES


def build_sensitivity_grid(d_array=None, delta_array=None):
    """Build 2D grid of composite sensitivity S_total(delta, d).

    S_total(delta, d) = S_LSPR(d) [nm/RIU] × I_ev(delta) [normalised]

    Parameters
    ----------
    d_array : array-like, optional
        AuNP diameters in nm. Defaults to np.arange(20, 81, 2).
    delta_array : array-like, optional
        Etch depths in µm. Defaults to np.linspace(0, 58.4, 50).

    Returns
    -------
    dict
        Keys: 'd_array', 'delta_array', 'S_LSPR', 'I_ev',
              'S_total_grid', 'FOM_total_grid',
              'optimal_d_nm', 'optimal_delta_um', 'S_total_max'
    """
    if d_array is None:
        d_array = np.arange(20, 81, 2)
    if delta_array is None:
        delta_array = np.linspace(0, 58.4, 50)

    d_array     = np.asarray(d_array, dtype=float)
    delta_array = np.asarray(delta_array, dtype=float)

    print(f"  Computing Mie sensitivity for {len(d_array)} diameters...")
    mie_data = sensitivity_vs_diameter(d_array)
    S_LSPR   = mie_data['sensitivity']   # shape: (n_d,)
    FOM_LSPR = mie_data['FOM']           # shape: (n_d,)

    print(f"  Computing TMM evanescent field for {len(delta_array)} etch depths...")
    tmm_df = sweep_etch_depth(delta_array)
    I_ev   = tmm_df['I_ev_normalised'].values   # shape: (n_delta,)

    # Broadcasting: S_grid[i, j] = I_ev[i] * S_LSPR[j]
    S_total_grid   = I_ev[:, np.newaxis] * S_LSPR[np.newaxis, :]    # (n_delta, n_d)
    FOM_total_grid = I_ev[:, np.newaxis] * np.nan_to_num(FOM_LSPR)[np.newaxis, :]

    flat_idx         = np.unravel_index(np.argmax(S_total_grid), S_total_grid.shape)
    optimal_delta_um = delta_array[flat_idx[0]]
    optimal_d_nm     = d_array[flat_idx[1]]
    S_total_max      = S_total_grid[flat_idx]

    return {
        'd_array':         d_array,
        'delta_array':     delta_array,
        'S_LSPR':          S_LSPR,
        'FOM_LSPR':        FOM_LSPR,
        'I_ev':            I_ev,
        'S_total_grid':    S_total_grid,
        'FOM_total_grid':  FOM_total_grid,
        'optimal_d_nm':    float(optimal_d_nm),
        'optimal_delta_um': float(optimal_delta_um),
        'S_total_max':     float(S_total_max),
    }


def find_optimal_design(grid_data):
    """Find global optimum and 90th-percentile design window.

    Parameters
    ----------
    grid_data : dict
        Output of build_sensitivity_grid().

    Returns
    -------
    dict
        Keys: 'optimal_d_nm', 'optimal_delta_um', 'S_total_max',
              'design_window_mask', 'sensitivity_at_optimal', 'FOM_at_optimal'
    """
    S = grid_data['S_total_grid']
    F = grid_data['FOM_total_grid']
    d = grid_data['d_array']
    delta = grid_data['delta_array']

    flat_idx = np.unravel_index(np.argmax(S), S.shape)
    threshold = np.nanpercentile(S, 90)
    mask = S >= threshold

    fom_at_opt = float(F[flat_idx]) if np.isfinite(F[flat_idx]) else np.nan

    return {
        'optimal_d_nm':          float(d[flat_idx[1]]),
        'optimal_delta_um':      float(delta[flat_idx[0]]),
        'S_total_max':           float(S[flat_idx]),
        'design_window_mask':    mask,
        'sensitivity_at_optimal': float(S[flat_idx]),
        'FOM_at_optimal':        fom_at_opt,
    }


def plot_sensitivity_map(grid_data=None,
                          save_path='figures/fig4_sensitivity_map'):
    """Two-panel composite sensitivity figure.

    Panel (a): S_LSPR vs AuNP diameter d from Mie theory.
    Panel (b): S_total vs remaining cladding thickness t_clad for four
               representative diameters, linear y-scale to clearly resolve
               the 1.47× sensitivity spread across diameters.

    Parameters
    ----------
    grid_data : dict, optional
        Output of build_sensitivity_grid(). Computed if None.
    save_path : str
        Base path without extension.

    Returns
    -------
    grid_data : dict
    """
    set_journal_style()
    if grid_data is None:
        grid_data = build_sensitivity_grid()

    d_array  = grid_data['d_array']
    S_LSPR   = grid_data['S_LSPR']
    opt_d    = grid_data['optimal_d_nm']
    S_max    = grid_data['S_total_max']

    # Diameters shown in panel (b)
    d_lines   = [20, 40, 60, 80]
    colors_b  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Back-calculate S_LSPR for each panel-b diameter
    mie_b = sensitivity_vs_diameter(np.array(d_lines, dtype=float))
    s_lspr_b = dict(zip(d_lines, mie_b['sensitivity']))

    # Compute S_total(t_clad) for each diameter
    t_clad_nm = np.linspace(0, 500, 501)                    # 0–500 nm remaining cladding
    etch_depths = np.maximum(_DELTA_MAX - t_clad_nm / 1000.0, 0.0)  # µm
    I_ev_arr = np.array([compute_evanescent_field(de)['I_ev_normalised']
                          for de in etch_depths])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['double_column'])
    fig.subplots_adjust(wspace=0.38)

    # ── Panel (a): S_LSPR vs d ──────────────────────────────────────────────
    ax1.plot(d_array, S_LSPR, color='#333333', lw=1.8, zorder=3)

    opt_d_idx = int(np.argmin(np.abs(d_array - opt_d)))
    s_opt = float(S_LSPR[opt_d_idx])
    ax1.scatter([opt_d], [s_opt], s=70, color='#d62728', zorder=5)

    # Annotate optimal point (offset to avoid crowding)
    x_off = (d_array.max() - d_array.min()) * 0.12
    y_off = (S_LSPR.max() - S_LSPR.min()) * 0.10
    ax1.annotate(
        f'$d^*$ = {opt_d:.0f} nm\n$S$ = {s_opt:.1f} nm RIU$^{{-1}}$',
        xy=(opt_d, s_opt),
        xytext=(opt_d + x_off, s_opt - y_off),
        fontsize=7, color='#d62728',
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8),
    )

    # Mark the four panel-b diameters with small ticks on the x-axis
    for d_val, col in zip(d_lines, colors_b):
        ax1.axvline(d_val, color=col, lw=0.6, ls=':', alpha=0.6)

    ax1.set_xlabel('AuNP diameter $d$ (nm)', fontsize=10)
    ax1.set_ylabel('$S_{LSPR}$ (nm RIU$^{-1}$)', fontsize=10)
    ax1.set_xlim(d_array.min() - 2, d_array.max() + 2)
    ax1.set_ylim(S_LSPR.min() * 0.97, S_LSPR.max() * 1.06)
    ax1.set_title('(a) Mie-theory LSPR sensitivity', fontsize=10, pad=6)

    # ── Panel (b): S_total vs t_clad ────────────────────────────────────────
    y_max = 0.0
    for d_val, col in zip(d_lines, colors_b):
        S_total_line = s_lspr_b[d_val] * I_ev_arr
        ax2.plot(t_clad_nm, S_total_line, lw=1.6, color=col,
                 label=f'$d$ = {d_val} nm')
        y_max = max(y_max, float(S_total_line[0]))

        # Annotate plateau value at the left edge (t_clad = 0)
        ax2.annotate(
            f'{S_total_line[0]:.1f}',
            xy=(2, S_total_line[0]),
            xytext=(2, S_total_line[0]),
            va='center', ha='left', fontsize=6.5, color=col,
        )

    # Shade active sensing window (I_ev > 1%, t_clad < ~500 nm but
    # set the visual cue at t_clad = 200 nm where I_ev ≈ 40–70%)
    ax2.axvspan(0, 200, alpha=0.07, color='gold', zorder=0)
    ax2.axvline(200, color='#8B6914', lw=0.9, ls='--', alpha=0.8)
    ax2.text(205, y_max * 0.52, '$t_{clad}$ = 200 nm\n(active window)',
             fontsize=6.5, color='#8B6914', va='center')

    ax2.set_xlabel('Remaining cladding $t_{clad}$ (nm)', fontsize=10)
    ax2.set_ylabel('$S_{total}$ (nm RIU$^{-1}$)', fontsize=10)
    ax2.set_title('(b) Composite sensitivity vs cladding', fontsize=10, pad=6)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, y_max * 1.12)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)

    save_figure(fig, save_path)
    plt.close(fig)
    return grid_data


def plot_fabrication_tolerance(grid_data=None,
                                save_path='figures/fig5_fabrication_tolerance'):
    """Show sensitivity tolerance to fabrication deviations around optimum.

    Parameters
    ----------
    grid_data : dict, optional
    save_path : str
    """
    set_journal_style()
    if grid_data is None:
        grid_data = build_sensitivity_grid()

    d      = grid_data['d_array']
    delta  = grid_data['delta_array']
    S      = grid_data['S_total_grid']
    opt_d  = grid_data['optimal_d_nm']
    opt_delt = grid_data['optimal_delta_um']
    S_max  = grid_data['S_total_max']

    opt_d_idx    = np.argmin(np.abs(d - opt_d))
    opt_delta_idx = np.argmin(np.abs(delta - opt_delt))

    S_vs_d     = S[opt_delta_idx, :]
    S_vs_delta = S[:, opt_d_idx]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column_tall'])

    # S vs d at optimal delta
    ax.plot(d, S_vs_d / S_max, '#1f77b4', lw=1.8, label=f'vs d at δ*={opt_delt:.1f} µm')
    # Tolerance bands (±10%, ±20% of diameter)
    for pct, alpha in [(10, 0.2), (20, 0.1)]:
        d_low  = opt_d * (1 - pct / 100)
        d_high = opt_d * (1 + pct / 100)
        ax.axvspan(d_low, d_high, alpha=alpha, color='#1f77b4',
                   label=f'±{pct}% d tolerance')

    # Secondary x-axis trick: plot S_vs_delta as a dashed line on normalised scale
    # We use a shared normalised scale on the same axis for clarity
    ax2 = ax.twiny()
    ax2.plot(delta, S_vs_delta / S_max, '#d62728', lw=1.8, ls='--',
             label=f'vs δ at d*={opt_d:.0f} nm')
    for pct, alpha in [(10, 0.2), (20, 0.1)]:
        d_low  = opt_delt * (1 - pct / 100)
        d_high = opt_delt * (1 + pct / 100)
        ax2.axvspan(d_low, d_high, alpha=alpha, color='#d62728')
    ax2.set_xlabel('Etch depth δ (µm)', fontsize=10, color='#d62728')
    ax2.tick_params(axis='x', colors='#d62728')

    ax.axhline(0.9, color='gray', ls=':', lw=1, label='−10% S level')
    ax.set_xlabel('AuNP diameter d (nm)', fontsize=10)
    ax.set_ylabel('Normalised $S_{total}$ / $S_{max}$', fontsize=10)
    ax.set_title('Fabrication tolerance analysis')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower center')

    save_figure(fig, save_path)
    plt.close(fig)


if __name__ == '__main__':
    gd = plot_sensitivity_map()
    plot_fabrication_tolerance(gd)
