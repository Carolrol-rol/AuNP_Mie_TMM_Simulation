"""Shared publication-quality matplotlib styling for all figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os

FIGURE_SIZES = {
    'single_column':      (3.5, 2.8),
    'single_column_tall': (3.5, 4.2),
    'double_column':      (7.0, 4.5),
    'double_column_tall': (7.0, 6.0),
}

ION_COLORS  = {'Cd2+': '#2ca02c', 'Pb2+': '#1f77b4', 'Hg2+': '#d62728'}
ION_MARKERS = {'Cd2+': 'o',       'Pb2+': 's',       'Hg2+': '^'}

_CB_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
             '#9467bd', '#8c564b', '#e377c2']


def set_journal_style():
    """Apply consistent journal-quality rcParams globally."""
    mpl.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans'],
        'axes.labelsize':    10,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'axes.titlesize':    11,
        'lines.linewidth':   1.5,
        'axes.linewidth':    0.8,
        'lines.markersize':  6,
        'figure.dpi':        300,
        'axes.prop_cycle':   mpl.cycler(color=_CB_CYCLE),
        'axes.grid':         True,
        'grid.linestyle':    ':',
        'grid.linewidth':    0.5,
        'grid.alpha':        0.5,
        'legend.frameon':    True,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '0.8',
        'figure.autolayout': True,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })


def save_figure(fig, base_path):
    """Save figure as both PDF and PNG at 300 dpi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    base_path : str
        Path without extension, e.g. 'figures/fig1_extinction'.
    """
    dir_name = os.path.dirname(base_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{base_path}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved: {base_path}.pdf and .png")
