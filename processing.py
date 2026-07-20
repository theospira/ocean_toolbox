import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d


def gauss_filter_z(ds, dvars=["temp", "psal"], sigma_dbar=2.0,
                   pres_dim="pres", min_frac=0.5, fill_gaps=False):
    """NaN-aware vertical Gaussian smoothing via normalized convolution.

    sigma_dbar : Gaussian std in dbar (grid-independent).
    min_frac   : drop points where < this fraction of kernel weight is valid data.
    fill_gaps  : if False, only originally-valid samples are returned; if True,
                 small interior gaps are also filled.
    """
    ax = ds[dvars[0]].get_axis_num(pres_dim)
    dz = float(np.abs(np.diff(ds[pres_dim].values)).mean())
    sigma = sigma_dbar / dz

    for d in dvars:
        a = ds[d].data.astype("float32", copy=False)
        m = np.isfinite(a)

        num = gaussian_filter1d(np.where(m, a, np.float32(0.0)), sigma, axis=ax,
                                mode="constant", cval=0.0)
        den = gaussian_filter1d(m.astype("float32"), sigma, axis=ax,
                                mode="constant", cval=0.0)

        valid = (den > min_frac) if fill_gaps else (m & (den > min_frac))
        np.divide(num, den, out=num, where=valid)
        num[~valid] = np.nan

        ds[d + "_sm"] = xr.DataArray(num, dims=ds[d].dims, coords=ds[d].coords)
    return ds