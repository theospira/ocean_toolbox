import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def no_nans(var):
    """Remove NaNs from a numpy array or xarray DataArray."""
    return var[np.isfinite(var)]


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def no_nans(arr):
    return arr[np.isfinite(arr)]


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def boxplot(
    var, dim=None, showfliers=False, ax=None, labels=None, figsize=None, **kwargs
):
    """
    Plot boxplots from xarray DataArray(s) or array-like inputs, excluding NaNs.

    Parameters
    ----------
    var : DataArray, array, or list of these
    dim : str or list of str, optional
    showfliers : bool, optional
    ax : matplotlib axis, array of axes, or tuple
    labels : list of str, optional
    figsize : tuple, optional
        Figure size (used only when creating new figure)
    **kwargs
    """

    # -----------------------------
    # Ensure list of variables
    # -----------------------------
    if not isinstance(var, (list, tuple)):
        var = [var]

    nvars = len(var)

    # -----------------------------
    # Handle axes input
    # -----------------------------
    if isinstance(ax, tuple):
        nrows, ncols = ax
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=figsize)
        axes = axes.flatten()

    elif ax is None:
        fig, axes = plt.subplots(1, nvars, figsize=figsize)
        if nvars == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

    else:
        axes = np.atleast_1d(ax).flatten()

    # -----------------------------
    # Check axis count
    # -----------------------------
    if len(axes) != nvars:
        raise ValueError(
            f"Number of axes ({len(axes)}) must match number of variables ({nvars})"
        )

    # -----------------------------
    # Loop through variables
    # -----------------------------
    for i, (v, ax_i) in enumerate(zip(var, axes)):

        if isinstance(v, xr.DataArray) and dim is not None:
            dims = [dim] if isinstance(dim, str) else dim
            reduced = v.transpose(..., *dims)
            other_dims = [d for d in reduced.dims if d not in dims]
            stacked = reduced.stack(sample=other_dims)

            cleaned = [
                no_nans(stacked.isel(sample=j).data.flatten())
                for j in range(stacked.sample.size)
            ]

            labels_i = (
                labels
                if labels is not None
                else [str(s) for s in stacked.sample.values]
            )

        else:
            arr = v.data if hasattr(v, "data") else v
            cleaned = [no_nans(arr.flatten())]
            labels_i = labels

        ax_i.boxplot(cleaned, showfliers=showfliers, labels=labels_i, **kwargs)

    return axes
