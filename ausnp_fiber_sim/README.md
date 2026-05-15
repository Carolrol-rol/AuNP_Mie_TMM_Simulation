# AuNP Fiber LSPR Sensor — Python Simulation Framework

## Paper

"Semi-Analytical Simulation of Evanescent Field Enhancement in Etched SMF-28 Fiber
with Gold Nanoparticles for Heavy Metal Ion Detection: A Mie Theory and Transfer
Matrix Parametric Study"
Target journal: *Sensors* (MDPI)

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

**Note for PyMieScatt users on SciPy >= 1.14**: If you see `ImportError: cannot import name 'trapz'`, patch PyMieScatt with:

```bash
sed -i 's/from scipy.integrate import trapz/from scipy.integrate import trapezoid as trapz/' \\
    $(python -c "import site; print(site.getsitepackages()\[1])")/PyMieScatt/Mie.py
sed -i 's/from scipy.integrate import trapz/from scipy.integrate import trapezoid as trapz/' \\
    $(python -c "import site; print(site.getsitepackages()\[1])")/PyMieScatt/Inverse.py
```

## Usage

```bash
cd ausnp\_fiber\_sim
python run\_all.py
```

Runtime: \~30 seconds on a modern laptop.

## Module overview

|Module|Physics|Key output|
|-|-|-|
|`gold\_dielectric.py`|Johnson \& Christy (1972) Au optical constants|epsilon(lambda)|
|`mie\_theory.py`|Exact Mie solution for spherical AuNPs (PyMieScatt)|S(d), FOM(d)|
|`transfer\_matrix.py`|Slab waveguide TMM for etched SMF-28|I\_ev(delta), l\_d(delta)|
|`sensitivity\_map.py`|Composite 2D parametric sensitivity|S\_total(delta, d)|
|`lod\_calculator.py`|LOD for Cd2+, Pb2+, Hg2+ vs WHO limits|LOD table|
|`validation.py`|Model validation vs published benchmarks|Pass/fail report|
|`plot\_style.py`|Shared journal-quality matplotlib style|--|

## Key simulation results

|Quantity|Value|
|-|-|
|Optimal AuNP diameter|80 nm|
|Optimal etch depth|58.4 µm|
|Max composite sensitivity|78.02 nm/RIU|
|LOD Cd2+|0.815 µg/L (WHO: 3) ✓|
|LOD Pb2+|1.897 µg/L (WHO: 10) ✓|
|LOD Hg2+|0.103 µg/L (WHO: 1) ✓|

## Figures generated

|Figure|Description|Size|
|-|-|-|
|`fig0\_gold\_optical\_constants`|J\&C (1972) n, k for gold (400-1000 nm)|Single col.|
|`fig1\_extinction\_spectra`|Mie extinction Q\_ext for d = 20-80 nm AuNPs|Single col.|
|`fig2\_sensitivity\_diameter`|LSPR sensitivity and FOM vs diameter|Single col.|
|`fig3\_evanescent\_field`|I\_ev and l\_d vs etch depth (log scale)|Single col.|
|`fig3b\_mode\_profile`|Schematic fiber cross-section at delta=30 µm|Single col.|
|`fig4\_sensitivity\_map`|**MAIN FIGURE** — 2D composite S\_total(delta, d)|Double col.|
|`fig5\_fabrication\_tolerance`|Sensitivity tolerance to fabrication deviations|Single col.|
|`fig6\_lod\_comparison`|LOD comparison vs WHO limits and literature|Double col.|
|`fig7\_sensitivity\_lod\_tradeoff`|LOD vs S\_total for design guidance|Single col.|
|`fig\_validation`|Model validation against 3 published datasets|Double col.|

## Physical model notes

### Mie theory (gold\_dielectric + mie\_theory)

* Uses bulk J\&C (1972) dielectric data (valid for d > 20 nm)
* Sub-20 nm particles require surface-damping corrections (not implemented)
* PyMieScatt provides exact Mie series solutions

### Transfer Matrix Method (transfer\_matrix)

* Simplified planar slab approximation of the cylindrical fiber
* LP01 mode propagation constant solved via characteristic equation
* Valid for gradual cladding removal (not deep-etch discontinuities)

### LOD calculation (lod\_calculator)

* Uses surface-effective dn/dc values calibrated from published sensors:

  * Cd2+/L-Cys: 53,000 RIU/(mol/L) \[Solomonea et al. 2022]
  * Pb2+/GSH: 42,000 RIU/(mol/L) \[Boruah and Biswas et al. 2018]
  * Hg2+/11-MUA: 750,000 RIU/(mol/L) \[Sadani et al. 2019]
* Instrument noise: sigma\_lambda = 0.01 nm (Ocean Insight USB4000)
* LOD = 3\*sigma\_lambda / S\_total (3-sigma criterion)

## Citation

If you use this code, please cite:
\[Author list] (2026). \[Paper title]. *Sensors*, XX(X), XXXX.
DOI: \[to be assigned upon publication]

Code repository: \[GitHub URL — to be added]

## License

MIT License — freely shareable as supplementary material.

