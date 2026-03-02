"""
Provides methods to visualize the information of data, giving a brief overview in figure.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from typing import Optional, Union, List, Dict
from pathlib import Path

from ..data import Any, to_numpy


def show_distribution(
    data        : Dict[str, Any],
    fn          : Union[str, Path],
    bins        : int  = 100,
    title       : str  = 'Data Distribution',
    axis_names  : List = ['Value', 'Frequency'],
    bounds      : Optional[List] = None,
    show_annots : bool = False,
    show_legend : bool = True,
):
    """
    Visualize the distribution of the data using histogram.

    ### Args
    - data: Dict
        - A dictionary mapping series names to their values.
    - fn: Union[str, Path]
        - File path for the saved figure.
    - bins: int, default 100
        - Number of bins in the histogram.
    - show_annots: bool, default False
        - Whether to annotate each data point with its value.
    - title: str, default 'Data Distribution'
        - The title of the figure.
    - axis_names: List, default ['Value', 'Frequency']
        - The names of the x and y axes.
    - bounds: Optional[List], default None
        - Left and right bounds of the histogram x-axis. *Must be a list of length 2 if provided.*
    - show_legend: bool, default True
        - Whether to show the legend.
    """
    labels = list(data.keys())
    data = np.stack([ to_numpy(x) for x in data.values() ], axis=0)
    assert data.ndim == 2, f"Data dimension should be 2, but got {data.ndim}."
    assert bounds is None or len(bounds) == 2, f"Bounds should be a list of length 2, but got {bounds}."
    # Preparation.
    N, K = data.shape
    data = data.transpose(1, 0)  # (K, N)
    # Plot.
    plt.hist(data, bins=bins, alpha=0.7, label=labels)
    if show_annots:
        for i in range(K):
            for j in range(N):
                plt.text(data[i, j], 0, f'{data[i, j]:.2f}', va='bottom', fontsize=6)
    plt.title(title)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    if show_legend:
        plt.legend()
    if bounds:
        plt.xlim(bounds)
    # Save.
    plt.savefig(fn)
    plt.close()


def show_performance(
    data        : Dict[str, Any],
    fn          : Union[str, Path],
    title       : str  = 'Performance',
    axis_names  : List = ['Value', 'Method'],
    show_annots : bool = True,
    show_legend : bool = False,
    bounds      : Optional[List] = None,
    fmt         : str  = '.2f',
):
    """
    Visualize performance comparison across methods using a horizontal bar chart.
    Each entry in `data` is drawn as one horizontal bar.

    ### Args
    - data: Dict[str, Any]
        - A dictionary mapping method/experiment names to their scalar performance values.
    - fn: Union[str, Path]
        - File path for the saved figure.
    - title: str, default 'Performance'
        - The title of the chart.
    - axis_names: List, default ['Value', 'Method']
        - The names of the x and y axes.
    - show_annots: bool, default True
        - Whether to annotate each bar with its value.
    - show_legend: bool, default False
        - Whether to show the legend.
    - bounds: Optional[List], default None
        - Left and right bounds of the x-axis. *Must be a list of length 2 if provided.*
    - fmt: str, default '.2f'
        - Format string for value annotations.
    """
    assert bounds is None or len(bounds) == 2, f"Bounds should be a list of length 2, but got {bounds}."

    if isinstance(fn, str):
        fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)

    labels = list(data.keys())
    values = to_numpy([v for v in data.values()])

    plt.figure(figsize=(8, max(2, 0.5 * len(labels))))
    ax = plt.gca()

    y_pos = np.arange(len(labels))
    for i, (label, v) in enumerate(zip(labels, values)):
        ax.barh(i, v, align='center', alpha=0.8, label=label)
        if show_annots:
            ax.text(v, i, f' {v:{fmt}}', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])

    if bounds:
        ax.set_xlim(bounds)

    if show_legend:
        ax.legend()

    plt.tight_layout()
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()


def show_history(
    data         : Dict[str, Any],
    fn           : Union[str, Path],
    data_pos     : Optional[Dict[str, Any]] = None,
    title        : str  = 'Data History',
    axis_names   : List = ['Time', 'Value'],
    ex_starts    : Dict[str, int] = {},
    show_annots  : bool  = False,
    show_legend  : bool  = True,
    show_scatter : bool  = True,
):
    """
    Visualize one or more value series as lines in a single figure.
    Multiple series in `data` are plotted together for comparison.

    ### Args
    - data: Dict[str, Any]
        - A dictionary mapping series names to their values.
    - fn: Union[str, Path]
        - File path for the saved figure.
    - data_pos: Optional[Dict[str, Any]], default None
        - A dictionary mapping series names to their x positions.
          If None, x positions are generated from `ex_starts` with step=1.
    - title: str, default 'Data History'
        - The title of the figure.
    - axis_names: List, default ['Time', 'Value']
        - The names of the x and y axes.
    - ex_starts: Dict[str, int], default {}
        - Starting x offset per series. *Only used when `data_pos` is None.*
    - show_annots: bool, default False
        - Whether to annotate each data point with its value.
    - show_legend: bool, default True
        - Whether to show the legend.
    - show_scatter: bool, default True
        - Whether to overlay scatter markers on data points.
    """
    # Make sure the fn's parent exists.
    if isinstance(fn, str):
        fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)

    # Preparation.
    history_name = list(data.keys())
    # Ensure data is numpy array for faster processing
    history_data = [np.array(x) for x in data.values()]
    N = len(history_name)
    Ls = [len(x) for x in history_data]
    Ss = [
            ex_starts[history_name[i]]
            if (history_name[i] in ex_starts.keys()) else 0
            for i in range(N)
        ]

    # Plot.
    # Pre-calculate figure to manage memory better
    plt.figure()

    for i in range(N):
        # Determine X coordinates efficiently
        if data_pos is None:
            cur_data_pos = np.arange(Ss[i], Ss[i]+Ls[i])
        else:
            cur_data_pos = data_pos[history_name[i]]

        plt.plot(
            cur_data_pos,
            history_data[i],
            label      = history_name[i],
            marker     = '.' if show_scatter else None,
            markersize = 4,
        )

    if show_annots:
        for i in range(N):
            # Recalculate X only if needed (logic consistent with above)
            if data_pos is None:
                cur_data_pos = np.arange(Ss[i], Ss[i]+Ls[i])
            else:
                cur_data_pos = data_pos[history_name[i]]

            cur_history_data = history_data[i]
            # Use zip for slightly faster iteration
            for x, y in zip(cur_data_pos, cur_history_data):
                plt.text(x, y, f'{y:.2f}', fontsize=6)

    plt.title(title)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    if show_legend:
        plt.legend()
    # Save.
    plt.savefig(fn)
    plt.close()