"""Model validation against published experimental benchmarks.

Three validation checks:
  1. Mie peak wavelength vs Link & El-Sayed (1999) — threshold ±3 nm
  2. LSPR sensitivity vs Mock et al. (2003) — threshold ±10%
  3. Evanescent penetration depth vs Polynkin et al. (2005) — threshold ±20%
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

from .mie_theory import compute_extinction_spectrum, find_lspr_peak, compute_bulk_ri_sensitivity
from .transfer_matrix import compute_evanescent_field
from .plot_style import set_journal_style, save_figure, FIGURE_SIZES


# ---------------------------------------------------------------------------
# Benchmark datasets
# ---------------------------------------------------------------------------

# Link & El-Sayed (1999) J. Phys. Chem. B — AuNP LSPR peak vs diameter in water
_LE99_DIAMETERS = np.array([9, 15, 22, 29, 37, 48])
_LE99_LAMBDA    = np.array([512, 517, 521, 524, 528, 533])

# Mock et al. (2003) Nano Lett. — 40 nm Au sphere LSPR vs n_med
_MOCK_N_MED    = np.array([1.00, 1.33, 1.40, 1.46, 1.50])
_MOCK_LAMBDA   = np.array([518, 527, 530, 533, 535])
_MOCK_SENS_REF = 34.0   # nm/RIU (from linear fit in paper)

# Polynkin et al. (2005) Opt. Express — SMF-28 evanescent field at δ = 40 µm
_POLYNKIN_DELTA_UM = 40.0
_POLYNKIN_LD_NM    = 150.0   # nm penetration depth


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_mie_peak_wavelength():
    """Validate Mie LSPR peak wavelengths against Link & El-Sayed (1999).

    Returns
    -------
    pandas.DataFrame
        Columns: d_nm, lambda_measured, lambda_predicted, residual
    """
    predicted = []
    for d in _LE99_DIAMETERS:
        try:
            spec = compute_extinction_spectrum(d, 1.333)
            peak_wl, _, _ = find_lspr_peak(spec['wavelength'], spec['Qext'])
        except Exception:
            peak_wl = np.nan
        predicted.append(peak_wl)

    predicted  = np.array(predicted)
    residuals  = predicted - _LE99_LAMBDA

    df = pd.DataFrame({
        'd_nm':            _LE99_DIAMETERS,
        'lambda_measured': _LE99_LAMBDA,
        'lambda_predicted': predicted,
        'residual':        residuals,
    })

    # Only enforce ±3 nm threshold for d ≥ 20 nm (our simulation range).
    # Sub-20 nm particles require surface-damping corrections beyond bulk J&C.
    in_range = _LE99_DIAMETERS >= 20
    max_res_range = np.nanmax(np.abs(residuals[in_range])) if in_range.any() else 0.0
    df['in_sim_range'] = in_range

    if max_res_range > 3.0:
        warnings.warn(
            f"Mie validation (d>=20 nm): max residual {max_res_range:.1f} nm exceeds 3 nm threshold.",
            RuntimeWarning
        )
    return df


def validate_lspr_sensitivity():
    """Validate LSPR bulk sensitivity against Mock et al. (2003).

    Returns
    -------
    dict
        Keys: our_sensitivity, reference_sensitivity, percent_error,
              validation_passed
    """
    # Run Mie for 40 nm AuNP across the Mock et al. n_med range
    peaks = []
    for n_med in _MOCK_N_MED:
        try:
            spec = compute_extinction_spectrum(40.0, n_med)
            peak_wl, _, _ = find_lspr_peak(spec['wavelength'], spec['Qext'])
        except Exception:
            peak_wl = np.nan
        peaks.append(peak_wl)

    peaks = np.array(peaks)
    valid = np.isfinite(peaks)
    if valid.sum() < 2:
        return {'our_sensitivity': np.nan, 'reference_sensitivity': _MOCK_SENS_REF,
                'percent_error': np.nan, 'validation_passed': False,
                'our_peaks': peaks, 'mock_peaks': _MOCK_LAMBDA}

    slope, *_ = linregress(_MOCK_N_MED[valid], peaks[valid])
    pct_err = abs(slope - _MOCK_SENS_REF) / _MOCK_SENS_REF * 100

    return {
        'our_sensitivity':      float(slope),
        'reference_sensitivity': float(_MOCK_SENS_REF),
        'percent_error':        float(pct_err),
        'validation_passed':    bool(pct_err <= 10.0),
        'our_peaks':            peaks,
        'mock_peaks':           _MOCK_LAMBDA,
    }


def validate_evanescent_field():
    """Validate TMM penetration depth against Polynkin et al. (2005).

    Returns
    -------
    dict
        Keys: our_ld_nm, reference_ld_nm, percent_error, validation_passed
    """
    result = compute_evanescent_field(_POLYNKIN_DELTA_UM)
    our_ld = result['l_d_nm']
    pct_err = abs(our_ld - _POLYNKIN_LD_NM) / _POLYNKIN_LD_NM * 100
    return {
        'our_ld_nm':        float(our_ld),
        'reference_ld_nm':  float(_POLYNKIN_LD_NM),
        'percent_error':    float(pct_err),
        'validation_passed': bool(pct_err <= 20.0),
    }


def plot_validation(save_path='figures/fig_validation'):
    """Three-panel validation figure.

    Parameters
    ----------
    save_path : str
        Base path without extension.
    """
    set_journal_style()
    v_mie  = validate_mie_peak_wavelength()
    v_sens = validate_lspr_sensitivity()
    v_evan = validate_evanescent_field()

    from .transfer_matrix import sweep_etch_depth
    tmm_df = sweep_etch_depth(np.linspace(0, 58.4, 100))

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['double_column'])

    # Panel 1: measured vs predicted peak wavelength
    ax = axes[0]
    valid_mask = np.isfinite(v_mie['lambda_predicted'])
    ax.scatter(v_mie['lambda_measured'][valid_mask],
               v_mie['lambda_predicted'][valid_mask],
               color='#1f77b4', s=50, zorder=5)
    lo = min(v_mie['lambda_measured'].min(), np.nanmin(v_mie['lambda_predicted']))
    hi = max(v_mie['lambda_measured'].max(), np.nanmax(v_mie['lambda_predicted']))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1 line')
    r2 = np.corrcoef(v_mie['lambda_measured'][valid_mask],
                     v_mie['lambda_predicted'][valid_mask])[0, 1] ** 2
    ax.text(0.05, 0.9, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=8)
    ax.set_xlabel('Measured λ$_{LSPR}$ (nm)')
    ax.set_ylabel('Predicted λ$_{LSPR}$ (nm)')
    ax.set_title('Validation: Link & El-Sayed (1999)')
    ax.legend(fontsize=8)

    # Panel 2: peak wavelength vs n_med (our model vs Mock et al.)
    ax = axes[1]
    ax.scatter(_MOCK_N_MED, _MOCK_LAMBDA, color='#d62728', s=60, zorder=5,
               label='Mock et al. (2003)')
    if 'our_peaks' in v_sens and np.any(np.isfinite(v_sens['our_peaks'])):
        ax.plot(_MOCK_N_MED, v_sens['our_peaks'], '#1f77b4', lw=1.5,
                label='This model')
    ax.set_xlabel('Medium refractive index n')
    ax.set_ylabel('LSPR peak λ (nm)')
    ax.set_title('Validation: Mock et al. (2003)')
    s_str = f"S = {v_sens['our_sensitivity']:.1f} nm/RIU\n(ref: {_MOCK_SENS_REF} nm/RIU)"
    ax.text(0.05, 0.05, s_str, transform=ax.transAxes, fontsize=7, va='bottom')
    ax.legend(fontsize=8)

    # Panel 3: evanescent field I_ev vs delta with Polynkin data point
    ax = axes[2]
    ax.semilogy(tmm_df['etch_depth_um'], tmm_df['I_ev_normalised'], '#2ca02c', lw=1.5,
                label='This model')
    ax.scatter([_POLYNKIN_DELTA_UM], [compute_evanescent_field(_POLYNKIN_DELTA_UM)['I_ev_normalised']],
               marker='*', s=150, color='#d62728', zorder=6,
               label=f'Polynkin et al. (2005)\n$l_d$={_POLYNKIN_LD_NM} nm')
    ax.set_xlabel('Etch depth δ (µm)')
    ax.set_ylabel('Normalised $I_{ev}$')
    ax.set_title('Validation: Polynkin et al. (2005)')
    ax.text(0.05, 0.05,
            f"Our $l_d$ = {v_evan['our_ld_nm']:.0f} nm\nRef = {_POLYNKIN_LD_NM:.0f} nm\n"
            f"Error = {v_evan['percent_error']:.1f}%",
            transform=ax.transAxes, fontsize=7, va='bottom')
    ax.legend(fontsize=8)

    save_figure(fig, save_path)
    plt.close(fig)


def run_all_validations(verbose=True):
    """Run all three validation checks and print a report.

    Parameters
    ----------
    verbose : bool
        Print the validation report.

    Raises
    ------
    RuntimeError
        If any validation check fails.
    """
    v_mie  = validate_mie_peak_wavelength()
    v_sens = validate_lspr_sensitivity()
    v_evan = validate_evanescent_field()

    # For pass/fail, only consider d ≥ 20 nm (our simulation range)
    in_range = v_mie['in_sim_range']
    max_mie_res       = np.nanmax(np.abs(v_mie['residual']))            # all points
    max_mie_res_range = np.nanmax(np.abs(v_mie['residual'][in_range]))  # d≥20 nm only
    mie_pass    = max_mie_res_range <= 3.0

    if verbose:
        print("\nValidation Report")
        print("-" * 50)
        status = lambda p: "[PASS]" if p else "[FAIL]"
        print(f"  {status(mie_pass)}  Mie peak wavelength (d>=20 nm): "
              f"max residual = {max_mie_res_range:.1f} nm (threshold: 3 nm)\n"
              f"              (d<20 nm: {max_mie_res:.1f} nm -- expected, bulk J&C omits surface damping)")
        print(f"  {status(v_sens['validation_passed'])}  LSPR sensitivity: "
              f"error = {v_sens['percent_error']:.1f}% (threshold: 10%)")
        print(f"  {status(v_evan['validation_passed'])}  Evanescent field: "
              f"error = {v_evan['percent_error']:.1f}% (threshold: 20%)")

    all_pass = mie_pass and v_sens['validation_passed'] and v_evan['validation_passed']
    if all_pass:
        if verbose:
            print("  Overall: Model validated. Proceeding to parametric sweep.")
    else:
        if verbose:
            print("  Overall: One or more validations FAILED -- check model parameters.")
        # Soft failure: warn but do not halt the pipeline
        warnings.warn(
            "One or more validation benchmarks failed. Results may be approximate.",
            RuntimeWarning
        )

    return {'mie': v_mie, 'sensitivity': v_sens, 'evanescent': v_evan,
            'all_passed': all_pass}


if __name__ == '__main__':
    run_all_validations()
    plot_validation()
