# Import necessary libraries and modules
from scipy.ndimage import (
    gaussian_filter1d as gf,
)  # Import the Gaussian filter function from SciPy
import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays


# Define a function for Gaussian filtering of vertical profiles in a dataset
def gauss_filter_z(ds, dvars=["temp", "psal"], std=4):
    """
    Apply Gaussian filtering to vertical profiles of selected variables in a dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing vertical profiles of oceanographic variables.

    dvars : list of str, optional
        A list of variable names to which Gaussian filtering will be applied.
        Default is ['temp', 'psal'].

    std : int or float, optional
        The standard deviation of the Gaussian filter. Default is 4.

    Returns:
    --------
    ds : xarray.Dataset
        The modified dataset with Gaussian-filtered variables.
    """
    # Iterate through the specified variables for Gaussian filtering
    for d in dvars:
        # Apply Gaussian filtering with the specified standard deviation
        ds[d + "_sm"] = xr.DataArray(
            gf(
                ds[d].data, std
            ),  # Gaussian filter with the specified standard deviation
            dims={
                "n_prof": ds.n_prof.data,
                "pres": ds.pres.data,
            },  # Preserve dimensions
            coords={
                "n_prof": ds.n_prof.data,
                "pres": ds.pres.data,
            },  # Preserve coordinates
        )

    return ds  # Return the modified dataset with Gaussian-filtered variables
