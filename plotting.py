import sys
import numpy as np
import pandas as pd
import xarray as xr
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import AutoDateLocator
import matplotlib.path as mpath

import cmocean.cm as cmo

import cartopy.crs as ccrs  # for plotting
import cartopy.feature as cfeature  # for map features

from warnings import filterwarnings as fw

fw("ignore")


def get_cb_colors(color_names=None, show=False):
    """
    Return colorblind-friendly hex colors, with optional preview plot.

    Parameters
    ----------
    color_names : list of str or None
        List of color names (e.g. ['blue', 'red']).
        If None, return full color cycle.

    show : bool, optional
        If True, display a plot of the selected colors.

    Returns
    -------
    list of str
        Hex color codes.
    """

    CB_color_map = {
        "blue": "#377eb8",
        "orange": "#ff7f00",
        "green": "#4daf4a",
        "pink": "#f781bf",
        "brown": "#a65628",
        "purple": "#984ea3",
        "grey": "#999999",
        "red": "#e41a1c",
        "yellow": "#dede00",
    }

    # Handle input
    if color_names is None:
        names = list(CB_color_map.keys())
        colors = list(CB_color_map.values())
    else:
        names = [c.lower() for c in color_names]
        colors = []
        for c in names:
            if c not in CB_color_map:
                raise ValueError(
                    f"Color '{c}' not recognised. Choose from {list(CB_color_map.keys())}"
                )
            colors.append(CB_color_map[c])

    # Optional plot
    if show:
        x = np.linspace(0, 10, 100)

        plt.figure(figsize=(6, 5))
        for i, (name, col) in enumerate(zip(names, colors)):
            plt.plot(x, (i + 1) * x, color=col, label=f"{name} ({col})")

        plt.title("Colorblind-friendly palette preview")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()

    return colors


def get_cmap(
    colors=[
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
    ][::-1]
):
    """
    Create and return a custom colormap for visualizations.

    This function generates a linear segmented colormap that transitions through a specified
    set of colors, ordered from dark red to dark blue. The colormap is defined using
    hexadecimal color codes and is intended for use in visualizations where a smooth
    transition between these colors is desired.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        A custom colormap object that can be used in plotting functions to map data values
        to colors.

    Notes
    -----
    - The colormap transitions through the following colors (in reverse order for the final colormap):
      '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
      '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'.
    - The resulting colormap can be applied to any data visualization that supports colormaps,
      such as those created with matplotlib.

    Example
    -------
    >>> cmap = get_cmap()
    >>> plt.imshow(data, cmap=cmap)
    >>> plt.colorbar()
    """

    # Define the colors in the order you've specified
    # Create a colormap that transitions from the first color to the last
    cmap_name = "custom_cmap"
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    return cmap


def circular_boundary(ax):
    """
    Create a circular boundary for a map plot.

    This function computes a circular boundary in axes coordinates, which can be used as a boundary
    for a map plot. It allows for panning and zooming, ensuring that the boundary remains circular.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to which the circular boundary will be applied.

    Returns:
    --------
    None

    Notes:
    ------
    The circular boundary is set using the `set_boundary` method of the provided axis.

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    >>> circular_boundary(ax)
    >>> ax.set_theta_zero_location('N')
    >>> ax.set_theta_direction(-1)
    >>> ax.set_rmax(1)
    >>> plt.show()
    """
    # Generate theta values for a complete circle
    theta = np.linspace(0, 2 * np.pi, 100)

    # Define the center and radius of the circle in axes coordinates
    center, radius = [0.5, 0.5], 0.5

    # Calculate the vertices of the circular path
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T

    # Create a matplotlib Path object representing the circular boundary
    circle = mpath.Path(verts * radius + center)

    # Set the circular boundary for the specified axis
    ax.set_boundary(circle, transform=ax.transAxes)


def circumpolar_figure(
    c=1,
    r=1,
    extent=None,
    ax_input=None,
    facecolor="darkgrey",
    continent_color="#ededed",
    coastline_color="k",
    coastline_resolution="110m",
    **fig_kwargs,
):
    """
    Create or configure a circumpolar map figure using a polar stereographic projection.

    Operates in two modes depending on whether existing axes are provided:

    - **Create mode** (``ax_input=None``): creates a new figure with ``r x c`` polar
      stereographic subplots.
    - **Replace mode** (``ax_input`` provided): replaces existing plain Axes with
      GeoAxes in-place, preserving the original subplot layout and any sharing.

    Parameters
    ----------
    c : int, optional
        Number of columns of subplots. Only used in create mode. Default is 1.
    r : int, optional
        Number of rows of subplots. Only used in create mode. Default is 1.
    extent : list of float, optional
        Map extent as ``[lon_min, lon_max, lat_min, lat_max]``.
        Must lie entirely within one hemisphere (all lats <= 0 or all lats >= 0).
        Default is ``[0, 360, -90, -45]`` (Southern Ocean).
    ax_input : matplotlib Axes, array of Axes, or None, optional
        Existing axes to replace with GeoAxes. If None, a new figure is created.
    facecolor : str, optional
        Background colour of the map (ocean/fill). Default is ``'darkgrey'``.
    continent_color : str, optional
        Fill colour for land/continent features. Default is ``'#ededed'``.
    coastline_color : str, optional
        Colour of coastline lines. Default is ``'k'`` (black).
    coastline_resolution : str, optional
        Resolution of Natural Earth coastlines. One of ``'110m'``, ``'50m'``,
        ``'10m'``. Default is ``'110m'``.
    **fig_kwargs
        Additional keyword arguments passed to ``plt.subplots()`` (create mode only),
        e.g. ``figsize``, ``constrained_layout``.

    Returns
    -------
    Create mode : tuple of (fig, ax, crs)
        - ``fig`` : matplotlib Figure
        - ``ax`` : GeoAxes or ndarray of GeoAxes, shaped ``(r, c)``
        - ``crs`` : ``cartopy.crs.PlateCarree`` instance for use in ``transform=`` calls

    Replace mode : tuple of (ax, crs)
        - ``ax`` : single GeoAxes or ndarray of GeoAxes matching the shape of ``ax_input``
        - ``crs`` : ``cartopy.crs.PlateCarree`` instance for use in ``transform=`` calls

    Raises
    ------
    ValueError
        If ``extent`` spans both hemispheres (i.e. ``lat_min < 0 < lat_max``).

    Examples
    --------
    Create a single Southern Ocean map from scratch:

    >>> fig, ax, crs = circumpolar_figure(figsize=(6, 6))

    Create a 2x3 grid of maps:

    >>> fig, axes, crs = circumpolar_figure(r=2, c=3, figsize=(14, 9))

    Replace axes in an existing mixed figure layout:

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> geo_axes, crs = circumpolar_figure(ax_input=axes)
    """

    # Default extent
    if extent is None:
        extent = [0, 360, -90, -45]
    min_lat, max_lat = extent[2], extent[3]

    # Determine projection
    if max_lat <= 0:
        projection = ccrs.SouthPolarStereo()
    elif min_lat >= 0:
        projection = ccrs.NorthPolarStereo()
    else:
        raise ValueError(
            "Extent spans both hemispheres. Circumpolar plots must be "
            "entirely Northern or Southern Hemisphere."
        )

    crs = ccrs.PlateCarree()

    # ── Case 1: existing axes passed in ──────────────────────────────────────
    if ax_input is not None:
        ax_array = np.atleast_1d(ax_input).ravel()

        # Replace each plain Axes with a GeoAxes at the same SubplotSpec position
        geo_axes = []
        for a in ax_array:
            fig = a.get_figure()
            subplotspec = a.get_subplotspec()  # preserves row/col/share layout
            a.remove()  # remove the plain Axes
            geo_ax = fig.add_subplot(subplotspec, projection=projection)
            geo_axes.append(geo_ax)

        geo_array = np.array(geo_axes).reshape(np.atleast_1d(ax_input).shape)

        for geo_ax in geo_axes:
            circular_boundary(geo_ax)
            geo_ax.set_extent(extent, crs=crs)
            geo_ax.add_feature(cfeature.LAND, zorder=9, facecolor=continent_color)
            geo_ax.coastlines(
                zorder=10, color=coastline_color, resolution=coastline_resolution
            )
            geo_ax.set_facecolor(facecolor)

        # Return same shape as input (scalar if single ax, array if slice)
        if geo_array.shape == (1,):
            return geo_array[0], crs
        return geo_array, crs

    # ── Case 2: create figure from scratch ───────────────────────────────────
    else:
        fig, ax = plt.subplots(
            r, c, subplot_kw={"projection": projection}, **fig_kwargs
        )
        ax_array = np.atleast_1d(ax).ravel()

        for a in ax_array:
            circular_boundary(a)
            a.set_extent(extent, crs=crs)
            a.add_feature(cfeature.LAND, zorder=9, facecolor=continent_color)
            a.coastlines(
                zorder=10, color=coastline_color, resolution=coastline_resolution
            )
            a.set_facecolor(facecolor)

        return fig, ax, crs


from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_gridlines(
    ax,
    alpha=0.75,
    draw_labels=False,
    lon_tick=None,
    lat_tick=None,
    linewidth=1,
    linestyle="--",
    zorder=6,
    **kwargs,
):
    """
    Add styled gridlines to a Cartopy GeoAxes.

    Parameters
    ----------
    ax : GeoAxes
        Cartopy axis.
    alpha : float
        Gridline transparency.
    draw_labels : bool
        Whether to draw coordinate labels.
    lon_tick : array-like, optional
        Longitude tick locations.
    lat_tick : array-like, optional
        Latitude tick locations.
    linewidth : float
    linestyle : str
    zorder : int
    **kwargs : dict
        Passed to ax.gridlines().
    """

    # Auto tick selection based on hemisphere
    if lon_tick is None:
        lon_tick = np.arange(-180, 181, 60)

    if lat_tick is None:
        extent = ax.get_extent(ccrs.PlateCarree())
        if extent[2] < 0:  # Southern Hemisphere
            lat_tick = np.arange(-80, -40, 10)
        else:  # Northern Hemisphere
            lat_tick = np.arange(50, 90, 10)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=draw_labels,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        xlocs=lon_tick,
        ylocs=lat_tick,
        **kwargs,
    )

    if draw_labels:
        gl.top_labels = True
        gl.right_labels = True
        gl.xlabel_style = {"rotation": 0, "fontsize": 10}
        gl.ylabel_style = {"rotation": 0, "fontsize": 10}
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
    else:
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        gl.right_labels = False

    return gl


def cyclic_rolling_mean(da, dim="lon", window=5, pad=20):
    """
    Apply rolling mean with cyclic padding along a dimension.

    Parameters
    ----------
    da : xr.DataArray
    dim : str
        Dimension to wrap (default: "lon")
    window : int
        Rolling window size
    pad : int
        Number of points to pad on each side

    Returns
    -------
    xr.DataArray
    """

    # pad cyclically
    da_pad = xr.concat(
        [da.isel({dim: slice(-pad, None)}), da, da.isel({dim: slice(0, pad)})], dim=dim
    )

    # apply rolling
    da_smooth = da_pad.rolling({dim: window}, center=True, min_periods=1).mean()

    # trim back to original size
    da_final = da_smooth.isel({dim: slice(pad, -pad)})

    return da_final


def fig_labels(
    a_x,
    a_y,
    ax,
    j=0,
    lower_case=True,
    label_format="({})",
    fs=15,
    add_bbox=False,
    bbox_fc=None,
    text=None,
    **kwargs,
):
    bbox_kw = None
    if add_bbox:
        if bbox_fc is None:
            bbox_fc = (
                ax.get_facecolor()
                if not isinstance(ax, (list, np.ndarray))
                else ax[0].get_facecolor()
            )
        bbox_kw = dict(
            facecolor=bbox_fc,
            edgecolor="None",
            boxstyle="round,pad=0.1",
            alpha=0.75,
        )
    alphabet = (
        [chr(i) for i in range(ord("a"), ord("z") + 1)]
        if lower_case
        else [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    )
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]
    if j + len(ax) > len(alphabet):
        raise ValueError(
            "The number of subplots exceeds the available alphabet letters."
        )
    if text is None:
        text_list = [None] * len(ax)
    elif isinstance(text, str):
        text_list = [text] * len(ax)
    else:
        if len(text) != len(ax):
            raise ValueError("`text` must have the same length as `ax`.")
        text_list = list(text)

    annotate_kw = dict(ha="center", va="center", fontweight="bold")
    annotate_kw.update(kwargs)

    for i, a in enumerate(ax):
        label = label_format.format(alphabet[j + i])
        if text_list[i] is not None:
            label = f"{label} {text_list[i]}"
        a.annotate(
            label,
            xy=(a_x, a_y),
            xycoords="axes fraction",
            fontsize=fs,
            bbox=bbox_kw,
            **annotate_kw,
        )
