# AuNP Fiber LSPR Sensor — Python Simulation Framework

## Paper

"Semi-Analytical Simulation of Evanescent Field Enhancement in Etched SMF-28 Fiber
with Gold Nanoparticles for Heavy Metal Ion Detection: A Mie Theory and Transfer
Matrix Parametric Study"

> Target journal: *Plasmonics* (Springer) — update citation block below upon acceptance.

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

**Note for PyMieScatt users on SciPy >= 1.14:** If you see `ImportError: cannot import name 'trapz'`, patch PyMieScatt with:

```bash
sed -i 's/from scipy.integrate import trapz/from scipy.integrate import trapezoid as trapz/' \
    $(python -c "import site; print(site.getsitepackages()[1])")/PyMieScatt/Mie.py

sed -i 's/from scipy.integrate import trapz/from scipy.integrate import trapezoid as trapz/' \
    $(python -c "import site; print(site.getsitepackages()[1])")/PyMieScatt/Inverse.py
```

## Usage

```bash
cd ausnp_fiber_sim
python run_all.py
```

Runtime: ~30 seconds on a modern laptop.

## Module Overview

| Module | Physics | Key Output |
|---|---|---|
| `gold_dielectric.py` | Johnson & Christy (1972) Au optical constants | epsilon(lambda) |
| `mie_theory.py` | Exact Mie solution for spherical AuNPs (PyMieScatt) | S(d), FOM(d) |
| `transfer_matrix.py` | Slab waveguide TMM for etched SMF-28 | I_ev(delta), l_d(delta) |
| `sensitivity_map.py` | Composite 2D parametric sensitivity | S_total(delta, d) |
| `lod_calculator.py` | LOD for Cd2+, Pb2+, Hg2+ vs WHO limits | LOD table |
| `validation.py` | Model validation vs published benchmarks | Pass/fail report |
| `plot_style.py` | Shared journal-quality matplotlib style | — |

## Key Simulation Results

| Quantity | Value |
|---|---|
| Optimal AuNP diameter | 80 nm |
| Optimal etch depth | 58.4 µm |
| Max composite sensitivity | 78.02 nm/RIU |
| LOD Cd2+ | 0.815 µg/L (WHO limit: 3 µg/L) ✓ |
| LOD Pb2+ | 1.897 µg/L (WHO limit: 10 µg/L) ✓ |
| LOD Hg2+ | 0.103 µg/L (WHO limit: 1 µg/L) ✓ |

## Figures Generated

| Figure | Description | Size |
|---|---|---|
| `fig0_gold_optical_constants` | J&C (1972) n, k for gold (400–1000 nm) | Single col. |
| `fig1_extinction_spectra` | Mie extinction Q_ext for d = 20–80 nm AuNPs | Single col. |
| `fig2_sensitivity_diameter` | LSPR sensitivity and FOM vs diameter | Single col. |
| `fig3_evanescent_field` | I_ev and l_d vs etch depth (log scale) | Single col. |
| `fig3b_mode_profile` | Schematic fiber cross-section at delta = 30 µm | Single col. |
| `fig4_sensitivity_map` | **MAIN FIGURE** — 2D composite S_total(delta, d) | Double col. |
| `fig5_fabrication_tolerance` | Sensitivity tolerance to fabrication deviations | Single col. |
| `fig6_lod_comparison` | LOD comparison vs WHO limits and literature | Double col. |
| `fig7_sensitivity_lod_tradeoff` | LOD vs S_total for design guidance | Single col. |
| `fig_validation` | Model validation against 3 published datasets | Double col. |

## Physical Model Notes

### Mie Theory (`gold_dielectric` + `mie_theory`)

- Uses bulk J&C (1972) dielectric data (valid for d ≥ 20 nm)
- Sub-20 nm particles require surface-damping corrections (not implemented)
- PyMieScatt provides exact Mie series solutions

### Transfer Matrix Method (`transfer_matrix`)

- Planar slab approximation of the cylindrical fiber geometry
- LP01 mode propagation constant solved via characteristic equation (Brent's method)
- Valid for gradual cladding removal; not intended for deep-etch discontinuities

### LOD Calculation (`lod_calculator`)

Surface-effective dn/dc values calibrated from published sensors:

| Ion | Ligand | dn/dc (RIU·L/mol) | Source |
|---|---|---|---|
| Cd2+ | L-Cystine | 53,000 | Şolomonea et al. (2022) |
| Pb2+ | Glutathione | 42,000 | Boruah & Biswas (2018) |
| Hg2+ | 11-Mercaptoundecanoic acid | 750,000 | Sadani et al. (2019) |

- Instrument noise: σ_λ = 0.01 nm (Ocean Insight USB4000)
- LOD = 3 × σ_λ / S_total (IUPAC 3-sigma criterion)

## Citation

If you use this code, please cite:

```
[Author list] (2026). [Paper title]. Plasmonics.
DOI: [to be assigned upon publication]
Zenodo: https://doi.org/10.5281/zenodo.20156295
```

## License

MIT License — freely shareable as supplementary material.
