import numpy as np
import xarray as xr
import gsw


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


def calc_mlp(ds, den_lim=0.03, ref_range=(10, 30)):
    """
    Compute mixed layer pressure (MLP) using a density threshold criterion.

    The mixed layer base is defined following de Boyer Montégut et al. (2004)
    as the first pressure level where the density exceeds the reference
    density by more than a threshold.

    The reference density is taken from the shallowest valid density
    measurement within a specified pressure range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing hydrographic profiles with pressure coordinate
        ``pres`` and density variable (``sigma0``, ``sig`` or ``rho``).

    den_lim : float, optional
        Density threshold (kg m⁻³) used to define the mixed layer base.
        Default = 0.03.

    ref_range : tuple, optional
        Pressure range (dbar) used to determine the reference density.
        Default = (10, 30).

    Returns
    -------
    xarray.Dataset
        Dataset with new variable ``mlp`` representing the mixed layer
        pressure (dbar).
    """

    # ------------------------------------------
    # Select density variable
    # ------------------------------------------
    if "sigma0" in ds:
        dens = ds["sigma0"]
    elif "sig" in ds:
        dens = ds["sig"]
    else:
        dens = ds["rho"]

    pres = ds["pres"]

    # ------------------------------------------
    # Reference density
    # ------------------------------------------
    dens_ref_layer = dens.sel(pres=slice(*ref_range))

    ref_idx = dens_ref_layer.notnull().argmax("pres").compute()

    dens_ref = dens_ref_layer.isel(pres=ref_idx)

    # ------------------------------------------
    # Search for threshold exceedance
    # ------------------------------------------
    dens_sub = dens.sel(pres=slice(ref_range[0], None))

    diff = np.abs(dens_sub - dens_ref)

    mask = diff > den_lim

    mlp_idx = mask.argmax("pres")

    valid = mask.any("pres")

    mlp_idx = mlp_idx.where(valid)

    mlp = dens_sub.pres.isel(pres=mlp_idx.fillna(0).astype("int64"))

    mlp = mlp.where(valid)

    ds["mlp"] = mlp

    # give attributes about method choices
    ds["mlp"].attrs["density_threshold"] = den_lim
    ds["mlp"].attrs["reference_range"] = ref_range

    return ds
