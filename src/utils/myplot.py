import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
import os
import numpy as np
from box import Box
from typing import List, Union

from ..utils.mymath import get_r_squared


def figure_grid(n_row, n_col, size=None, size_ax=None, gridspec_kw=dict()):
    if size is None and size_ax is not None:
        size = np.array([n_col, n_row]) * size_ax
    fig, axs = plt.subplots(n_row, n_col, figsize=size, constrained_layout=True, gridspec_kw=gridspec_kw)
    return fig, axs


def figure_save(fig:Figure, path, dpi=100, save_svg=False, pad_inches=0.04):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    if save_svg:
        root, _ = os.path.splitext(path)
        path_svg = f"{root}.pdf"
        fig.savefig(path_svg, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    plt.close(fig)


def convert_axis_units(ax:Axes, axis='x', scale_factor=1, format_spec='d'):
    assert axis in ['x', 'y'], "axis must be either 'x' or 'y'"

    if axis == 'x':
        locs = ax.get_xticks()
        ax.set_xticks(locs)
        if format_spec.endswith('d'):
            ax.set_xticklabels([f'{round(loc*scale_factor):{format_spec}}' for loc in locs])
        else:
            ax.set_xticklabels([f'{loc*scale_factor:{format_spec}}' for loc in locs])
    else:  # y-axis
        locs = ax.get_yticks()
        ax.set_yticks(locs)
        if format_spec.endswith('d'):
            ax.set_yticklabels([f'{round(loc*scale_factor):{format_spec}}' for loc in locs])
        else:
            ax.set_yticklabels([f'{loc*scale_factor:{format_spec}}' for loc in locs])


def parse_axes_shape(axs: np.ndarray) -> tuple[int, int]:
    shape = axs.shape
    
    if len(shape) == 2:
        return shape  # 2D array, shape is already (n_row, n_col)
    
    if len(shape) == 1:
        n = shape[0]
        if n == 1:
            return 1, 1  # Single Axes in a 1D array
        
        ax0 = axs[0]
        ax1 = axs[1]
        
        if ax1.get_position().y0 < ax0.get_position().y0:
            return n, 1  # Column of Axes
        else:
            return 1, n  # Row of Axes
    
    raise ValueError("Unexpected Axes array shape")



def set_labels_subplot(axs: Union[np.ndarray, Axes], xl, yl):
    if type(axs) is Axes:
        assert type(xl) is str and type(yl) is str
        axs.set_xlabel(xl)
        axs.set_ylabel(yl)
    elif type(axs) is np.ndarray:
        n_row, n_col = parse_axes_shape(axs)
        assert n_row == len(yl) and n_col == len(xl)
        if n_row == 1:
            for ax, l in zip(axs, xl):
                ax.set_xlabel(l)
            axs[0].set_ylabel(yl[0])
        elif n_col == 1:
            for ax, l in zip(axs, yl):
                ax.set_ylabel(l)
            axs[-1].set_xlabel(yl[0])
        else:
            for r in range(n_row):
                axs[r][0].set_ylabel(yl[r])
            for c in range(n_col):
                axs[-1][c].set_xlabel(xl[c])
                

def set_ticks_subplot(axs: Union[np.ndarray, Axes]):
    pass


def unify_axis_limits(axs: Union[List[Axes], np.ndarray, set], axis='y'):
    """
    Unify the x or y axis limits across multiple Axes instances.
    
    Args:
        axs: List, set, or array of Axes instances
        axis: 'x' or 'y' to specify which axis to unify
    """
    assert axis in ['x', 'y'], "axis must be either 'x' or 'y'"
    
    # Convert to list if needed
    if isinstance(axs, set):
        axs = list(axs)
    elif isinstance(axs, np.ndarray):
        axs = axs.flatten()
    
    # Get all current limits
    if axis == 'x':
        limits = [ax.get_xlim() for ax in axs]
    else:
        limits = [ax.get_ylim() for ax in axs]
    
    # Find common min and max
    common_min = min(lim[0] for lim in limits)
    common_max = max(lim[1] for lim in limits)
    common_limit = (common_min, common_max)
    
    # Apply to all axes
    for ax in axs:
        if axis == 'x':
            ax.set_xlim(common_limit)
        else:
            ax.set_ylim(common_limit)


def draw_bar_plot(ax:Axes, mean_list, std_list=None, **kwargs):
    settings = Box(dict(
        bar_width = 0.8,
        edge_margin = 0.25,     # Between bar and ax edge

        xlabel = None,
        ylabel = None,
        xticks = None,
        yticks = None,
    ))
    settings.update(kwargs)

    if std_list is not None:
        assert len(mean_list) == len(std_list)

    x = np.arange(len(mean_list)) + 0.5
    ax.set_xlim(0, len(mean_list))

    if "colors" in settings:
        ax.bar(x, mean_list, width=settings.bar_width, color=settings.colors)
    else:
        ax.bar(x, mean_list, width=settings.bar_width)

    ax.set_ylabel(settings.ylabel)
    ax.set_xticks(x, settings.xticks)

    



def draw_r2_plot(ax:Axes, xdata, ydata, **kwargs):
    """
    kwargs
        edge_margin: float. Decides the edge margin of the plot
        full_fit_label: bool. Decides whether to include the equation in the label
        
        base_lw: y=x line's linewidth
        base_ls: y=x line's linestyle
        base_lc: y=x line's linecolor
    """
    settings = Box(dict(
        edge_margin = 0.1,
        full_fit_label = False,
        title = None,
        title_fontsize = None,

        min_val = None,
        max_val = None,

        reg_lw = 2.5,
        reg_lc = 'red',
        reg_mc = 'black',
        reg_ma = 0.3,
        reg_ms = 5,

        base_lw = 1,
        base_ls = '--',
        base_lc = 'gray',

        xlabel = "X value",
        ylabel = "Y value",
        lb_fontsize = 10,

        legend_fontsize = 8,
        legend_pos = "lower right",

        grid_lw = 0.5,
        grid_lc = 'gray',
        grid_la = 0.5,
        grid_ls = '--',
        aspect = 'equal'
    ))
    settings.update(kwargs)

    result = dict()

    rsq_value = get_r_squared(xdata, ydata)
    result["R2"] = rsq_value

    if settings.min_val is not None:
        min_val = settings.min_val
    else:
        min_val = min(min(xdata), min(ydata))

    if settings.max_val is not None:
        max_val = settings.max_val
    else:
        max_val = max(max(xdata), max(ydata))

    dist_val = (max_val - min_val) * settings.edge_margin

    y_fit = np.polyfit(xdata, ydata, 1)
    if settings.full_fit_label:
        label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={rsq_value:.2f}$"
    else:
        label = f"$R^2={rsq_value:.2f}$"
    
    sns.regplot(
        x=xdata,
        y=ydata,
        scatter_kws={"color": settings.reg_mc, "alpha": settings.reg_ma, "s": settings.reg_ms},
        line_kws={"color": settings.reg_lc, "lw": settings.reg_lw},
        ax=ax
    )
    ax.plot(
        [min_val - dist_val, max_val + dist_val],
        [min_val - dist_val, max_val + dist_val],
        linewidth=1 if "base_lw" not in kwargs else kwargs["base_lw"],
        color="gray" if "base_lc" not in kwargs else kwargs["base_lc"],
        linestyle="--" if "base_ls" not in kwargs else kwargs["base_ls"],
    )
    ax.set_xlabel(settings.xlabel, fontsize=settings.lb_fontsize)
    ax.set_ylabel(settings.ylabel, fontsize=settings.lb_fontsize)
    ax.set_xlim([min_val - dist_val, max_val + dist_val])
    ax.set_ylim([min_val - dist_val, max_val + dist_val])
    ax.legend([
        Line2D([0], [0], color=settings.reg_lc, lw=settings.reg_lw)], 
        [label,], 
        fontsize=settings.legend_fontsize, 
        loc=settings.legend_pos
    )
    ax.grid(linestyle=settings.grid_ls, linewidth=settings.grid_lw, color=settings.grid_lc, alpha=settings.grid_la)

    if settings.aspect is not None:
        ax.set_aspect(settings.aspect)
    
    if settings.title is not None:
        if settings.title_fontsize is not None:
            ax.set_title(settings.title, fontsize=settings.title_fontsize)
        else:
            ax.set_title(settings.title)

    return result

