"""Master runner script for the AuNP fiber LSPR sensor simulation.

Executes all physics modules in sequence and prints a formatted results summary.
"""

import os
import sys
import time
import traceback
import io

import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows to allow Greek/special chars in print output
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure src/ is importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

HEADER = """
================================================
 AuNP Fiber LSPR Sensor - Simulation Framework
 Semi-Analytical Model (Mie Theory + TMM)
 Paper: Sensors (MDPI) - 2026
================================================
"""


def _step(n, total, label):
    print(f"[{n}/{total}] {label}...", flush=True)
    return time.perf_counter()


def _done(t0):
    print(f"        Done in {time.perf_counter() - t0:.1f} s\n", flush=True)


def main():
    print(HEADER)
    os.makedirs('figures', exist_ok=True)
    results = {}

    TOTAL = 6   # steps including validation

    # ------------------------------------------------------------------
    # Step 0: Validation
    # ------------------------------------------------------------------
    t0 = _step(0, TOTAL, "Validating model against published benchmarks")
    try:
        from src.validation import run_all_validations, plot_validation
        val = run_all_validations(verbose=True)
        plot_validation()
        results['validation'] = val
    except Exception as e:
        print(f"  [WARNING] Validation step encountered an error: {e}")
        print("  Continuing with parametric sweep.\n")
    _done(t0)

    # ------------------------------------------------------------------
    # Step 1: Gold optical constants
    # ------------------------------------------------------------------
    t0 = _step(1, TOTAL, "Loading gold optical constants (Johnson & Christy 1972)")
    try:
        from src.gold_dielectric import load_jc_data, plot_jc_data
        jc_df = load_jc_data()
        print(f"  Loaded {len(jc_df)} data points, "
              f"{jc_df['wavelength_nm'].min():.0f}-{jc_df['wavelength_nm'].max():.0f} nm")
        plot_jc_data()
    except Exception as e:
        print(f"  [ERROR] Gold dielectric module failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    _done(t0)

    # ------------------------------------------------------------------
    # Step 2: Mie theory
    # ------------------------------------------------------------------
    t0 = _step(2, TOTAL, "Running Mie theory parametric sweep (d = 20-80 nm)")
    try:
        from src.mie_theory import (sensitivity_vs_diameter,
                                    plot_extinction_spectra,
                                    plot_sensitivity_vs_diameter)
        mie_data = sensitivity_vs_diameter(np.arange(20, 81, 5))
        plot_extinction_spectra()
        plot_sensitivity_vs_diameter()

        opt_fom_idx = np.nanargmax(mie_data['FOM'])
        results['mie'] = {
            'optimal_d_nm':       float(mie_data['diameters'][opt_fom_idx]),
            'sensitivity_nm_RIU': float(mie_data['sensitivity'][opt_fom_idx]),
            'FOM':                float(mie_data['FOM'][opt_fom_idx]),
            'FWHM_nm':            float(mie_data['FWHM'][opt_fom_idx]),
        }
        print(f"  Optimal FOM diameter: {results['mie']['optimal_d_nm']:.0f} nm, "
              f"S = {results['mie']['sensitivity_nm_RIU']:.1f} nm/RIU, "
              f"FOM = {results['mie']['FOM']:.2f}")
    except Exception as e:
        print(f"  [ERROR] Mie theory module failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    _done(t0)

    # ------------------------------------------------------------------
    # Step 3: Transfer Matrix Method
    # ------------------------------------------------------------------
    t0 = _step(3, TOTAL, "Running Transfer Matrix Method (delta = 0-58.4 um)")
    try:
        from src.transfer_matrix import (sweep_etch_depth,
                                          plot_evanescent_field,
                                          plot_mode_profile)
        tmm_df = sweep_etch_depth()
        plot_evanescent_field()
        plot_mode_profile(etch_depth_um=30.0)
        results['tmm'] = {
            'max_I_ev':    float(tmm_df['I_ev_normalised'].max()),
            'ld_at_40um':  float(tmm_df.loc[
                (tmm_df['etch_depth_um'] - 40).abs().idxmin(), 'l_d_nm']),
        }
        print(f"  l_d at delta=40 um: {results['tmm']['ld_at_40um']:.1f} nm, "
              f"max I_ev: {results['tmm']['max_I_ev']:.4f}")
    except Exception as e:
        print(f"  [ERROR] Transfer matrix module failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    _done(t0)

    # ------------------------------------------------------------------
    # Step 4: 2D sensitivity map
    # ------------------------------------------------------------------
    t0 = _step(4, TOTAL, "Building 2D composite sensitivity map")
    try:
        from src.sensitivity_map import (build_sensitivity_grid,
                                          find_optimal_design,
                                          plot_sensitivity_map,
                                          plot_fabrication_tolerance)
        grid_data = build_sensitivity_grid()
        opt       = find_optimal_design(grid_data)
        plot_sensitivity_map(grid_data)
        plot_fabrication_tolerance(grid_data)
        results['sensitivity_map'] = opt
        print(f"  Optimal: delta* = {opt['optimal_delta_um']:.1f} um, "
              f"d* = {opt['optimal_d_nm']:.0f} nm, "
              f"S* = {opt['S_total_max']:.2f} nm/RIU, "
              f"FOM = {opt.get('FOM_at_optimal', float('nan')):.2f}")
    except Exception as e:
        print(f"  [ERROR] Sensitivity map module failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    _done(t0)

    # ------------------------------------------------------------------
    # Step 5: LOD calculation
    # ------------------------------------------------------------------
    t0 = _step(5, TOTAL, "Computing limits of detection (Cd2+, Pb2+, Hg2+)")
    try:
        from src.lod_calculator import (compute_all_lods,
                                         plot_lod_comparison,
                                         plot_sensitivity_lod_tradeoff,
                                         get_ion_parameters)
        S_total = results['sensitivity_map']['S_total_max']
        lod_df  = compute_all_lods(S_total)
        plot_lod_comparison(S_total)
        plot_sensitivity_lod_tradeoff()
        results['lod'] = lod_df
    except Exception as e:
        print(f"  [ERROR] LOD calculator module failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    _done(t0)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    _print_summary(results)
    _save_csv(results)


def _print_summary(results):
    opt  = results.get('sensitivity_map', {})
    lod  = results.get('lod', pd.DataFrame())
    ion_pars = {}
    try:
        from src.lod_calculator import get_ion_parameters
        ion_pars = get_ion_parameters()
    except Exception:
        pass

    sep = "=" * 62
    print(f"\n{sep}")
    print("  SIMULATION RESULTS SUMMARY")
    print(sep)
    print("  Optimal design point:")
    print(f"    AuNP diameter:        d* = {opt.get('optimal_d_nm', 0):.0f} nm")
    print(f"    Cladding etch depth:  delta* = {opt.get('optimal_delta_um', 0):.1f} um")
    print(f"    Max sensitivity:      S* = {opt.get('S_total_max', 0):.2f} nm/RIU")
    fom = opt.get('FOM_at_optimal', float('nan'))
    print(f"    Figure of Merit:      FOM = {fom:.2f}" if fom == fom else
          "    Figure of Merit:      FOM = N/A")

    print("\n  Predicted limits of detection:")
    print(f"    {'Ion':<8} {'LOD (ug/L)':<14} {'WHO limit (ug/L)':<20} Status")
    print(f"    {'-'*54}")
    for ion in ['Cd2+', 'Pb2+', 'Hg2+']:
        if ion in lod.index:
            row = lod.loc[ion]
            who = ion_pars.get(ion, {}).get('who_limit_ug_L', '?')
            status = 'FAIL' if row['exceeds_who_limit'] else 'PASS'
            print(f"    {ion:<8} {row['LOD_ug_L']:<14.3f} {who:<20} {status}")
        else:
            print(f"    {ion:<8} {'N/A':<14} {'N/A':<20} N/A")

    print(f"\n  Figures saved to: ./figures/")
    figs = [
        'fig0_gold_optical_constants.pdf/.png',
        'fig1_extinction_spectra.pdf/.png',
        'fig2_sensitivity_diameter.pdf/.png',
        'fig3_evanescent_field.pdf/.png',
        'fig3b_mode_profile.pdf/.png',
        'fig4_sensitivity_map.pdf/.png  <- KEY FIGURE',
        'fig5_fabrication_tolerance.pdf/.png',
        'fig6_lod_comparison.pdf/.png',
        'fig7_sensitivity_lod_tradeoff.pdf/.png',
        'fig_validation.pdf/.png',
    ]
    for f in figs:
        print(f"    {f}")
    print(sep)


def _save_csv(results):
    try:
        opt = results.get('sensitivity_map', {})
        lod = results.get('lod', pd.DataFrame())

        rows = [
            {'Parameter': 'optimal_d_nm',     'Value': opt.get('optimal_d_nm', np.nan)},
            {'Parameter': 'optimal_delta_um',  'Value': opt.get('optimal_delta_um', np.nan)},
            {'Parameter': 'S_total_max_nm_RIU','Value': opt.get('S_total_max', np.nan)},
            {'Parameter': 'FOM_at_optimal',    'Value': opt.get('FOM_at_optimal', np.nan)},
        ]
        if not lod.empty:
            for ion in lod.index:
                rows.append({'Parameter': f'LOD_{ion}_ug_L',
                             'Value':     lod.loc[ion, 'LOD_ug_L']})

        pd.DataFrame(rows).to_csv('results_summary.csv', index=False)
        print("\n  Results saved to: results_summary.csv")
    except Exception as e:
        print(f"  [WARNING] Could not save results CSV: {e}")


if __name__ == '__main__':
    main()
