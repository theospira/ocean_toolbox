

import xarray as xr
import numpy as np
import pandas as pd
import gsw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
import openpyxl
import matplotlib.dates as mdates


### Load SI data ###

def load_monthly_si_data(base_url="https://noaadata.apps.nsidc.org/NOAA/G02135/south/monthly/data"):
    """
    Load NSIDC monthly Southern Hemisphere sea ice extent/area data into an xarray Dataset.

    Downloads and concatenates the 12 monthly CSVs (S_01_extent_v4.0.csv ... S_12_extent_v4.0.csv)
    from the NSIDC Sea Ice Index v4.0 monthly data archive:
    https://noaadata.apps.nsidc.org/NOAA/G02135/south/monthly/data/

    Parameters
    ----------
    base_url : str
        Base URL of the NSIDC monthly data directory. Defaults to the Southern Hemisphere
        archive; pass the "north" equivalent for Arctic data.

    Returns
    -------
    xr.Dataset
        Dataset indexed by monthly `time`, with data variables:
        - extent : sea ice extent (10^6 km^2)
        - area   : sea ice area (10^6 km^2)
        NSIDC's -9999 missing-data sentinel values are masked to NaN.

    Notes
    -----
    Each row also carries a `source_dataset` field (e.g. NSIDC-0051, NSIDC-0803) marking
    a sensor/product transition in the record; this is dropped from the returned Dataset
    but is present in the raw CSVs if needed.
    """
    datasets = []
    for mm in range(1, 13):
        url = f"{base_url}/S_{mm:02d}_extent_v4.0.csv"
        df = pd.read_csv(url)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"mo": "month"})
        df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
        df = df.set_index("time")[["extent", "area", "source_dataset", "region"]]
        datasets.append(df)

    df_all = pd.concat(datasets).sort_index()

    # Mask NSIDC's -9999 missing-data sentinel
    df_all[["extent", "area"]] = df_all[["extent", "area"]].mask(df_all[["extent", "area"]] < 0)

    ds = df_all[["extent", "area"]].to_xarray()
    ds["extent"].attrs["units"] = "10^6 km^2"
    ds["area"].attrs["units"] = "10^6 km^2"
    ds = ds.sortby("time")
    ds.attrs["description"] = "NSIDC Sea Ice Index v4.0 monthly Southern Hemisphere sea ice extent and area"

    return ds


def load_daily_si_data(url):
    """
    Load an NSIDC daily sea ice extent CSV into an xarray Dataset.

    Reads one of NSIDC's daily Sea Ice Index v4.0 files (e.g.
    S_seaice_extent_daily_v4.0.csv / N_seaice_extent_daily_v4.0.csv), builds a
    daily datetime index from the year/month/day columns, and drops the
    source-product metadata column.

    Parameters
    ----------
    url : str
        URL or path to a daily NSIDC extent CSV
        (south/daily/data/S_seaice_extent_daily_v4.0.csv or the north equivalent).

    Returns
    -------
    xr.Dataset
        Dataset indexed by daily `time`, with data variables:
        - extent  : sea ice extent (10^6 km^2)
        - missing : missing-data flag/value for that day
    """
    # Read
    df = pd.read_csv(url, skiprows=1)
    # Rename columns to something sane
    df = df.rename(columns={
        "YYYY": "year",
        "    MM": "month",
        "  DD": "day",
        " 10^6 sq km": "extent",
        " 10^6 sq km.1": "missing"
    })
    # Drop the useless source column
    df = df.drop(columns=[col for col in df.columns if "Source data product" in col])
    # Build time index
    df["time"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("time")
    # Keep only what matters
    df = df[["extent", "missing"]]
    # Convert to xarray
    ds = df.to_xarray()
    # Add attrs (optional but nice)
    ds["extent"].attrs["units"] = "10^6 km^2"
    ds["missing"].attrs["description"] = "missing data flag"
    return ds

def get_monthly_std():
    """
    Compute monthly standard deviation of daily NH/SH sea ice extent anomalies.

    Loads NSIDC daily Southern and Northern Hemisphere extent data (1980-01-01
    to 2026-01-01), computes daily climatology and anomaly for each hemisphere
    via `add_daily_climatology_and_anomaly`, then bins the anomalies by
    calendar month and takes the standard deviation within each bin.

    Requires `load_daily_si_data` and `add_daily_climatology_and_anomaly` to be
    defined/imported.

    Returns
    -------
    xr.Dataset
        Dataset indexed by monthly `time` (month-start, 1980-01 to 2025-12),
        with data variables for the monthly std of daily anomalies:
        - nh_extent (and its anomaly-derived variables)
        - sh_extent (and its anomaly-derived variables)
        exact variable names depend on what `add_daily_climatology_and_anomaly`
        adds to each hemisphere's Dataset before merging.
    """
    # daily data
    url_sh = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v4.0.csv"
    url_nh = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v4.0.csv"
    t_slice = slice("1980-01-01", "2026-01-01") # adjust this if necessary. included this line for my own purposes
    ds_nh = load_daily_si_data(url_nh)[['extent']].sel(time=t_slice)
    ds_sh = load_daily_si_data(url_sh)[['extent']].sel(time=t_slice)
    ds_nh = ds_nh.rename({"extent": "nh_extent"})
    ds_sh = ds_sh.rename({"extent": "sh_extent"})
    std_sh = ds_sh.groupby_bins("time", bins=pd.date_range("1980-01-01", "2026-01-01", freq="MS")).std().rename({"time_bins": "time"})
    std_nh = ds_nh.groupby_bins("time", bins=pd.date_range("1980-01-01", "2026-01-01", freq="MS")).std().rename({"time_bins": "time"})
    std_m = xr.merge([std_sh, std_nh])
    std_m['time'] = pd.date_range("1980-01-01", "2026-01-01", freq="MS")[:-1]
    return std_m

# -----------------------------------------------------------------
        ### load Roemmich and Gilson Argo Climatology ###
# -----------------------------------------------------------------

import gzip
import shutil
import tempfile
from io import BytesIO
import requests

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _download_gunzip_open(url, engine="h5netcdf", decode_times=False):
    """
    Download a .nc.gz file and open as xarray.Dataset.
    """
    r = requests.get(url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        with gzip.GzipFile(fileobj=BytesIO(r.content)) as gz:
            shutil.copyfileobj(gz, tmp)
        tmp.flush()
        ds = xr.open_dataset(tmp.name, engine=engine, decode_times=decode_times)
        ds.load()  # must load before tmp file is deleted on context exit
    return ds


def _set_time_from_filename(ds, yyyymm):
    """Force correct time coordinate from filename (e.g. '201901')."""
    time = pd.to_datetime(yyyymm, format="%Y%m")
    if "TIME" in ds:
        ds = ds.rename({"TIME": "time"})
        ds["time"] = ("time", [time])
    else:
        ds = ds.expand_dims(time=[time])
    return ds


def _generate_urls(base_url, start, end):
    dates = pd.date_range(start=start, end=end, freq="MS")
    return [
        f"{base_url}RG_ArgoClim_{d.strftime('%Y%m')}_2019.nc.gz"
        for d in dates
    ]


def _spatial_mask(ds, lat_min, lat_max, lon_min, lon_max):
    """Boolean mask instead of .sel(slice) since lat/lon aren't guaranteed monotonic."""
    mask = xr.ones_like(ds.LATITUDE, dtype=bool)
    if lat_min is not None:
        mask = mask & (ds.LATITUDE >= lat_min)
    if lat_max is not None:
        mask = mask & (ds.LATITUDE <= lat_max)
    ds = ds.sel(LATITUDE=mask)

    if lon_min is not None or lon_max is not None:
        lmask = xr.ones_like(ds.LONGITUDE, dtype=bool)
        if lon_min is not None:
            lmask = lmask & (ds.LONGITUDE >= lon_min)
        if lon_max is not None:
            lmask = lmask & (ds.LONGITUDE <= lon_max)
        ds = ds.sel(LONGITUDE=lmask)
    return ds


# -----------------------------------------------------------------
# Main loader
# -----------------------------------------------------------------

def load_argo_rg(
    start=None,
    end=None,
    lat_min=-90,
    lat_max=-50,      # Southern Ocean default
    lon_min=None,
    lon_max=None,
    pres_min=None,
    pres_max=None,
    time_min=None,
    time_max=None,
    base_url="https://sio-argo.ucsd.edu/RG/",
    compute_absolute=True,
):
    """
    Load Roemmich-Gilson Argo climatology (2004-2019) + monthly updates,
    sliced to a lat/lon/pressure/time subset. Default provides Southern 
    Ocean circumpolar dataset (south of -50°).

    Parameters
    ----------
    start, end : str or None
        Monthly update range to fetch, "YYYY-MM". If None, defaults to
        the full available range: updates start 2019-01 (climatology
        covers 2004-01 to 2018-12 as static mean fields), and end
        defaults to the current month.
    lat_min, lat_max, lon_min, lon_max : float or None
        Bounding box. Defaults to south of -50 (Southern Ocean). Pass
        lat_max=-55 etc. to reproduce a tighter SO slice.
    pres_min, pres_max : float or None
        Pressure range, applied after load via .sel(PRESSURE=slice(...)).
    time_min, time_max : str or None
        Time range applied after merge, e.g. "2020-01", "2022-12".
    base_url : str
    compute_absolute : bool
        If True, convert anomalies to absolute values and compute
        TEOS-10 derived fields (SA, CT, rho).

    Returns
    -------
    xarray.Dataset with variables: ctemp, asal, pres, lat, lon, time
    (+ rho if compute_absolute).

    Examples
    --------
    >>> ds = load_argo_rg()                                    # full 2004-present, default south of -50°
    >>> ds = load_argo_rg(lat_max=-55)                         # tighter SO slice
    >>> ds = load_argo_rg(start="2020-01", end="2022-12")      # custom update range
    >>> ds = load_argo_rg(pres_min=0, pres_max=500)            # upper 500 dbar only
    """
    if start is None:
        start = "2019-01"
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m")

    # --- climatology (static mean fields, 2004-01 base) ---
    ds_temp = _download_gunzip_open(base_url + "RG_ArgoClim_Temperature_2019.nc.gz")
    ds_sal = _download_gunzip_open(base_url + "RG_ArgoClim_Salinity_2019.nc.gz")
    ds_clim = xr.merge([ds_temp, ds_sal])

    base = pd.Timestamp("2004-01-01")
    ds_clim = ds_clim.rename({"TIME": "time"})
    ds_clim["time"] = [base + pd.DateOffset(months=int(m)) for m in ds_clim.time.values]
    ds_clim = _spatial_mask(ds_clim, lat_min, lat_max, lon_min, lon_max)

    # --- monthly updates ---
    datasets = []
    for url in _generate_urls(base_url, start, end):
        try:
            ds = _download_gunzip_open(url)
            yyyymm = url.split("_")[2]
            ds = _set_time_from_filename(ds, yyyymm)
            ds = _spatial_mask(ds, lat_min, lat_max, lon_min, lon_max)
            datasets.append(ds)
        except Exception as e:
            print(f"Skipping {url}: {e}")

    ds_updates = xr.concat(datasets, dim="time") if datasets else None

    # --- merge ---
    ds = xr.concat([ds_clim, ds_updates], dim="time") if ds_updates is not None else ds_clim

    # --- pressure slice ---
    if pres_min is not None or pres_max is not None:
        ds = ds.sel(PRESSURE=slice(pres_min, pres_max))

    # --- time slice ---
    if time_min is not None or time_max is not None:
        ds = ds.sel(time=slice(time_min, time_max))

    # --- rename ---
    rename_dict = {"LATITUDE": "lat", "LONGITUDE": "lon"}
    if "ARGO_TEMPERATURE_MEAN" in ds:
        rename_dict["ARGO_TEMPERATURE_MEAN"] = "ctemp"
    if "ARGO_SALINITY_MEAN" in ds:
        rename_dict["ARGO_SALINITY_MEAN"] = "asal"
    if "PRESSURE" in ds:
        rename_dict["PRESSURE"] = "pres"
    ds = ds.rename(rename_dict)

    if not compute_absolute:
        return ds

    # --- anomaly -> absolute + TEOS-10 ---
    # NOTE: [0] indexing assumes the first time step holds the reference
    # mean field used to de-anomalize. Needs double-checking this 
    # still holds once pres/time slicing is applied above 
    # (e.g. if time_min excludes the reference step).
    ds_aa = ds.copy()
    ds_aa["sal"] = ds_aa.ARGO_SALINITY_ANOMALY + ds_aa.asal[0].values
    ds_aa["temp"] = ds_aa.ARGO_TEMPERATURE_ANOMALY + ds_aa.ctemp[0].values
    ds_aa = ds_aa.drop_vars(
        ["ARGO_SALINITY_ANOMALY", "ARGO_TEMPERATURE_ANOMALY", "asal", "ctemp"]
    )

    ds_aa["asal"]  = gsw.SA_from_SP(ds_aa.sal, ds_aa.pres, ds_aa.lon, ds_aa.lat)
    ds_aa["ctemp"] = gsw.CT_from_t(ds_aa.asal, ds_aa.temp, ds_aa.pres)
    # ds_aa["rho"]   = gsw.rho(ds_aa.asal, ds_aa.ctemp, ds_aa.pres)

    return ds_aa



