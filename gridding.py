import pandas as pd
import numpy as np
import xarray as xr
from tqdm.auto import tqdm


from scipy.stats import mode


def mode_gridding_ts(dsvar):

    t_bins = pd.date_range("2003-12-31", "2022-01-01", freq="1M")

    if "pres" in dsvar.dims:
        arr = np.ndarray([t_bins.size - 1, 360, 40, ds.pres.size]) * np.nan
    else:
        arr = np.ndarray([t_bins.size - 1, 360, 40]) * np.nan

    # define lon min and max resp
    gs = 1
    lon_min = -180
    lon_max = 180
    lon = np.arange(lon_min, lon_max + gs, gs)
    lon_labels = range(0, lon_max - lon_min, gs)

    lat_min = -80
    lat_max = -40
    lat = np.arange(lat_min, lat_max + gs, gs)
    lat_labels = range(0, lat_max - lat_min, gs)

    # group by seasons
    var = dsvar.groupby_bins(dsvar.time, bins=t_bins, labels=np.arange(len(t_bins) - 1))

    # group into lon bins
    for t, gr_t in tqdm(var):
        var1 = gr_t.groupby_bins("lon", lon, labels=lon_labels, restore_coord_dims=True)

        # now group into lat bins for each lon group:
        for ln, gr_ln in var1:
            var2 = gr_ln.groupby_bins(
                "lat", lat, labels=lat_labels, restore_coord_dims=True
            )

            # now take the mode!
            for lt, gr_lt in var2:
                arr[t, ln, lt] = mode(gr_lt, nan_policy="omit")[0]

    return arr


def add_bathym_to_ds(ds):
    """
    Add bathymetry data to an input dataset by interpolating it onto the
    dataset's (lat, lon) grid.

    The bathymetry field is loaded from GEBCO, subset to the Southern Ocean,
    and interpolated to match the input dataset coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing ``lat`` and ``lon`` coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with an added ``bth`` DataArray (lat, lon) representing
        bathymetry (depth, typically negative values).

    Notes
    -----
    - Uses linear interpolation via ``xarray.interp``.
    - Preserves bathymetric gradients better than coarsening.
    - Assumes ``ds.lat`` and ``ds.lon`` are 1D coordinates.
    """

    # load bathymetry
    bth = xr.open_mfdataset("/albedo/work/user/thspir002/data/GEBCO/gebco_2025-40S.nc")

    # ensure coordinate names match
    # (GEBCO usually already uses lat/lon, but just in case)
    if "latitude" in bth.coords and "lat" not in bth.coords:
        bth = bth.rename({"latitude": "lat", "longitude": "lon"}, errors="ignore")

    # select Southern Ocean
    bth = bth.elevation.sel(lat=slice(-80, -40))

    # optional: handle lon convention mismatch
    if ds.lon.max() > 180:
        bth = bth.assign_coords(lon=((bth.lon + 360) % 360)).sortby("lon")

    # interpolate onto ds grid
    bth_interp = bth.interp(lat=ds.lat, lon=ds.lon, method="linear")

    # assign
    ds["bth"] = bth_interp

    return ds


def add_mdt_to_ds(ds):
    """
    Add mean dynamic topography (MDT) to an input dataset by interpolating it
    onto the dataset's (lat, lon) grid.

    The MDT field is loaded from a CNES dataset, optionally subset to the
    Southern Ocean, and interpolated to match the input dataset coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing ``lat`` and ``lon`` coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with an added ``mdt`` DataArray (lat, lon) interpolated onto
        the input grid.

    Notes
    -----
    - Uses linear interpolation via ``xarray.interp``.
    - Assumes ``ds.lat`` and ``ds.lon`` are 1D coordinates.
    - MDT is time-mean; the first time index is used.
    """

    # load MDT
    mdt = xr.open_dataset(
        "/albedo/home/thspir002/code/projects/00-sandbox/"
        "cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_1993-2012-40S.nc"
    )

    # rename to match ds
    if "latitude" in mdt.coords and "lat" not in mdt.coords:
        mdt = mdt.rename({"latitude": "lat", "longitude": "lon"})

    # select Southern Ocean if relevant
    mdt = mdt.sel(lat=slice(-80, None))

    # take time mean (single time anyway)
    mdt_field = mdt.mdt.isel(time=0)

    # interpolate onto ds grid
    mdt_interp = mdt_field.interp(lat=ds.lat, lon=ds.lon, method="linear")

    # assign to dataset
    ds["mdt"] = mdt_interp

    return ds
