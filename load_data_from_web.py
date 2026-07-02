

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