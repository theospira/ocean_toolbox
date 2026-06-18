from scipy.stats import pearsonr, t
import numpy as np
import xarray as xr
import gsw


def compute_correlation(
    da1: xr.DataArray, da2: xr.DataArray, smoothed: int = None
) -> tuple[float, float]:
    """
    Compute the Pearson correlation coefficient and (optionally corrected) p-value
    between two 1D or 2D xarray DataArrays without a 'time' dimension.

    Parameters:
    da1 (xr.DataArray): First input DataArray.
    da2 (xr.DataArray): Second input DataArray.
    smoothed (int or None): If set (e.g. 10 for a 10° smoothing), correct the p-value
                             assuming the effective sample size is reduced by this factor.

    Returns:
    tuple[float, float]: Pearson correlation coefficient and p-value (corrected if smoothed is given).
    """
    # Flatten the arrays and mask NaNs
    arr1 = da1.values.ravel()
    arr2 = da2.values.ravel()
    mask = ~np.isnan(arr1) & ~np.isnan(arr2)

    # Check for sufficient data
    if mask.sum() < 3:
        return np.nan, np.nan

    x = arr1[mask]
    y = arr2[mask]
    r, p_raw = pearsonr(x, y)

    # If not smoothed, return raw correlation and p-value
    if smoothed is None:
        return r, p_raw

    # Adjust degrees of freedom based on smoothing
    n_eff = len(x) / smoothed
    df_eff = max(int(n_eff) - 2, 1)
    t_stat = r * np.sqrt(df_eff / (1 - r**2))
    p_corr = 2 * (1 - t.cdf(np.abs(t_stat), df=df_eff))

    return r, p_corr


def compute_weighted_average(ds, var=None, dims=("lat", "lon")):
    """
    Compute the area-weighted average of one or more variables over specified dimensions.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input data. If a Dataset, must contain the variable(s) and a grid cell area field ('gca').
        If a DataArray, the area is calculated dynamically using `calc_grid_cell_area`.

    var : str or list of str or None
        Variable name(s) to compute the weighted average for (ignored if `ds` is a DataArray).
        Required if `ds` is a Dataset.

    dims : tuple of str
        Dimensions over which to average. Default is ('lat', 'lon').

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The area-weighted average as a DataArray (if one variable) or Dataset (if multiple).
    """

    if isinstance(ds, xr.DataArray):
        da = ds
        # Compute grid cell area on the same grid as da
        gca = calc_grid_cell_area(da)
        da = da.where(np.isfinite(da))
        weights = gca.where(np.isfinite(da))
        return (da * weights).sum(dims) / weights.sum(dims)
    else:
        if var is None:
            raise ValueError("If ds is a Dataset, 'var' must be specified.")
        if "gca" not in ds:
            ds = calc_grid_cell_area(ds)
        # Handle single variable or list of variables
        if isinstance(var, str):
            var_list = [var]
        else:
            var_list = [v for v in list(var) if v != "gca"]
        results = {}
        for v in var_list:
            dsvar = ds[v]
            dsvar = dsvar.where(np.isfinite(dsvar))
            weights = ds.gca.where(np.isfinite(dsvar))
            results[v] = (dsvar * weights).sum(dims) / weights.sum(dims)
        # Return a DataArray if only one variable, else a Dataset
        if len(results) == 1:
            return list(results.values())[0]
        else:
            return xr.Dataset(results)


def compute_weighted_std(ds, var=None, dims=("lat", "lon")):
    """
    Compute the area-weighted standard deviation of a variable in the dataset or a DataArray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset containing the variable of interest and grid cell area ('gca'), or a DataArray.
    var : str or None
        Name of the variable for which the weighted standard deviation is computed (if ds is a Dataset).
        If ds is a DataArray, this is ignored.
    dims : tuple
        Dimensions over which to compute the standard deviation (default: ('lat', 'lon')).

    Returns
    -------
    xarray.DataArray
        Area-weighted standard deviation of the specified variable.
    """

    if isinstance(ds, xr.DataArray):
        da = ds
        gca = calc_grid_cell_area(da)
        da = da.where(np.isfinite(da))
        weights = gca.where(np.isfinite(da))
        mean = (da * weights).sum(dims) / weights.sum(dims)
        return np.sqrt(((weights * (da - mean) ** 2).sum(dims)) / weights.sum(dims))
    else:
        if var is None:
            raise ValueError("If ds is a Dataset, 'var' must be specified.")
        if "gca" not in ds:
            ds = calc_grid_cell_area(ds)
        dsvar = ds[var]
        dsvar = dsvar.where(np.isfinite(dsvar))
        weights = ds.gca.where(np.isfinite(dsvar))
        mean = (dsvar * weights).sum(dims) / weights.sum(dims)
        return np.sqrt(((weights * (dsvar - mean) ** 2).sum(dims)) / weights.sum(dims))


from scipy import stats


def polyfit_xr(dsvar, dim="time", deg=1):
    """
    Polynomial fitting for xarray DataArray along a specified dimension.
    If dim is 'time', converts to numeric (nanoseconds → years) before fitting
    so that slope units are per year rather than per nanosecond.

    Parameters
    ----------
    dsvar : xarray.DataArray
    dim   : str   dimension along which to fit (default 'time')
    deg   : int   polynomial degree (default 1)

    Returns
    -------
    slope       : xarray.DataArray   slope (units/yr if time, else units/dim-unit)
    intercept   : xarray.DataArray   intercept
    p_value     : xarray.DataArray   p-value of the slope (only for deg=1)
    r_squared   : xarray.DataArray   R² of the fit
    """

    # ── time → numeric (years) ───────────────────────────────────────────────
    if dim == "time":
        t_numeric = xr.DataArray(
            (dsvar[dim].values.astype("datetime64[ns]").astype("float64"))
            / (1e9 * 86400 * 365.25),  # nanoseconds → years
            coords={dim: dsvar[dim]},
            dims=[dim],
        )
        da = dsvar.assign_coords({dim: t_numeric.values})
        fit_dim = dim
    else:
        da = dsvar
        fit_dim = dim

    # ── polyfit ──────────────────────────────────────────────────────────────
    pf = da.polyfit(fit_dim, deg=deg, cov=True)
    slope = (
        pf.polyfit_coefficients.sel(degree=deg - 0, drop=True)
        if deg > 1
        else pf.polyfit_coefficients[0]
    )
    intercept = pf.polyfit_coefficients[-1]

    # R² via xr.polyval
    fitted = xr.polyval(da[fit_dim], pf.polyfit_coefficients)
    residuals = da - fitted
    ss_res = (residuals**2).sum(dim=fit_dim)
    ss_tot = ((da - da.mean(dim=fit_dim)) ** 2).sum(dim=fit_dim)
    r_squared = 1 - ss_res / ss_tot

    # ── p-value (deg=1 only, via scipy t-test) ───────────────────────────────
    if deg == 1:
        n = da.sizes[fit_dim]
        se = np.sqrt(pf.polyfit_covariance[0, 0])  # std error of slope
        t_stat = slope / se
        p_value = xr.apply_ufunc(
            lambda t: 2 * stats.t.sf(np.abs(t), df=n - 2),
            t_stat,
            dask="parallelized",
            output_dtypes=[float],
        )
    else:
        p_value = None

    return slope, intercept, p_value, r_squared


def compute_mean_da(da):
    """
    Calculates the climatological monthly mean of a DataArray and returns a new DataArray
    with the corresponding monthly mean value at each time step.

    Parameters:
    - da (xarray.DataArray): Input DataArray with a 'time' coordinate.

    Returns:
    - xarray.DataArray: DataArray of the same shape and coordinates as input,
                        with each time step replaced by its climatological monthly mean.
    """

    clim_mean = da.groupby("time.month").mean("time")
    arr = np.full(da.shape, np.nan)

    # Get axis indices
    time_axis = da.get_axis_num("time")
    non_time_axes = [i for i, d in enumerate(da.dims) if d != "time"]

    # Iterate over time steps
    for t, month in enumerate(da["time"].dt.month.values):
        # Use tuple to select all elements along non-time axes, and t for time axis
        slc = [slice(None)] * arr.ndim
        slc[time_axis] = t
        arr[tuple(slc)] = clim_mean.sel(month=month).data

    mean_da = xr.DataArray(arr, dims=da.dims, coords=da.coords).where(da.notnull())

    return mean_da


def calc_climatology(x, v=None, output=None):
    """
    Compute monthly climatology and/or anomalies.

    This function calculates a monthly climatology based on the time dimension
    and optionally returns:
        - the climatological mean (monthly mean, broadcast to time)
        - anomalies relative to that climatology

    Supports both xarray Dataset and DataArray inputs.

    Parameters
    ----------
    x : xr.Dataset or xr.DataArray
        Input data containing a 'time' dimension.
    v : str, optional
        Variable name (required if `x` is a Dataset).
        Ignored if `x` is a DataArray.
    output : {"both", "mn", "anm"}, optional
        Specifies what to return:
            - "both" : climatology and anomaly
            - "mn"   : climatology only
            - "anm"  : anomaly only

        Defaults:
            - DataArray input  -> "anm"
            - Dataset input    -> "both"

    Returns
    -------
    xr.Dataset or xr.DataArray
        If input is:
        - Dataset: returns Dataset with added variables:
            <v>_mn  : monthly climatology (broadcast to time)
            <v>_anm : anomaly
        - DataArray:
            returns DataArray or tuple depending on `output`:
                - "anm"  -> anomaly DataArray
                - "mn"   -> climatology DataArray
                - "both" -> (climatology, anomaly)

    Notes
    -----
    - Climatology is computed as the mean over all years for each calendar month.
    - The climatology is broadcast back onto the original time dimension using
      groupby indexing.
    - Assumes a standard datetime-like 'time' coordinate.
    """

    # Handle defaults
    if isinstance(x, xr.DataArray):
        da = x
        if output is None:
            output = "anm"
    else:
        if v is None:
            raise ValueError("Must provide variable name `v` when input is a Dataset.")
        da = x[v]
        if output is None:
            output = "both"

    # Monthly climatology (month dimension only)
    clim = da.groupby("time.month").mean("time")

    # Broadcast climatology onto full time dimension
    clim_full = clim.sel(month=da["time.month"]).drop("month")

    # Compute anomaly
    anm = da - clim_full

    # Return logic
    if isinstance(x, xr.DataArray):
        if output == "mn":
            return clim_full
        elif output == "anm":
            return anm
        elif output == "both":
            return clim_full, anm
        else:
            raise ValueError("output must be one of {'both', 'mn', 'anm'}")

    else:  # Dataset
        if output in ["both", "mn"]:
            x[v + "_mn"] = clim_full
        if output in ["both", "anm"]:
            x[v + "_anm"] = anm
        return x


def calc_grid_cell_area(ds):
    """
    Calculate the area of each grid cell in a given dataset or DataArray using precomputed distances.

    Args:
        ds (xarray.Dataset or xarray.DataArray): Input dataset or dataarray containing latitude (and optionally longitude) coordinates.

    Returns:
        xarray.Dataset or xarray.DataArray: With a new variable 'gca' representing grid cell areas (if Dataset),
                                            or a DataArray of grid cell areas (if DataArray).
    """
    lat = ds.lat.data

    if "lon" in ds.dims or "lon" in ds.coords:
        lon = ds.lon.data
        lat = ds.lat.data - 0.5  # Adjust latitude for grid center alignment

        # Extend longitude to define grid cell edges
        lon = np.arange(lon[0] - 0.5, lon[-1] + 1.5, 1)

        # Create meshgrid
        lnlt = np.meshgrid(lon, lat)

        # Compute zonal (east-west) and meridional (north-south) distances
        dist_x = gsw.distance(lnlt[0], lnlt[1]) / 1e3  # km
        dist_y = gsw.distance(lnlt[1], lnlt[0]) / 1e3  # km

        # Calculate grid cell areas (km²)
        grid_cell_areas = dist_x * dist_y

        # Return appropriately based on input type
        if isinstance(ds, xr.Dataset):
            ds = ds.copy()
            ds["gca"] = xr.DataArray(grid_cell_areas, dims=("lat", "lon"))
            return ds
        else:
            return xr.DataArray(
                grid_cell_areas,
                dims=("lat", "lon"),
                coords={"lat": ds.lat, "lon": ds.lon},
            )

    else:
        # Only latitude dimension; assume 1° longitude bin per cell
        lat = ds.lat.data - 0.5  # Same alignment as above
        lon = np.array([0, 1])  # 1° longitude strip

        # Create meshgrid for one longitude strip (arbitrary longitude, e.g. 0 to 1)
        lnlt = np.meshgrid(lon, lat)

        # Compute distances as above
        dist_x = gsw.distance(lnlt[0], lnlt[1]) / 1e3  # km (zonal width per lat)
        dist_y = gsw.distance(lnlt[1], lnlt[0]) / 1e3  # km (meridional height per lat)

        # Area for each latitude band
        grid_cell_areas = dist_x * dist_y  # shape: (len(lat), 1)
        grid_cell_areas = grid_cell_areas.flatten()  # shape: (len(lat),)

        if isinstance(ds, xr.Dataset):
            ds = ds.copy()
            ds["gca"] = xr.DataArray(
                grid_cell_areas, dims=("lat",), coords={"lat": ds.lat}
            )
            return ds
        else:
            return xr.DataArray(grid_cell_areas, dims=("lat",), coords={"lat": ds.lat})
