"""
This script will iterate through the magnetometer dataset and try and redownload missing sections. If it can't get
the section after a specified number of tries, it should print out the interval so it can be manually inspected on
the SuperMAG website

stuff to look for:
    - Missing stretches
"""

import xarray as xr
import numpy as np
from data.supermag_download import getDataForInterval
import pandas as pd

mag_fn_pattern = "mag_data/mag_data_{}.nc"
save_fn_pattern = "E:/mag_data_{}.nc"
max_tries = 3

for yr in range(1990, 2019):
    print(yr)
    year = str(yr)
    # gather magnetometer data for the year
    mag_file = mag_fn_pattern.format(year)
    dataset = xr.open_dataset(mag_file)
    dates = pd.to_datetime(dataset.Date_UTC.values)

    if dates[0] != pd.Timestamp(year=yr, month=1, day=1, hour=0, minute=0, second=0):
        start = pd.Timestamp(year=yr, month=1, day=1, hour=0, minute=0, second=0)
        interval = dates[0] - start
        duration = interval.total_seconds()
        hours = int(np.floor(duration / 3600))
        minutes = int(np.floor((duration - hours * 3600) / 60))
        data_for_interval = getDataForInterval(start, hours, minutes)
        dataset = xr.merge([dataset, data_for_interval])

    # First fill in missing stretches of data
    time_diffs = (np.diff(dataset.Date_UTC.values) / 1e9).astype(int)  # in seconds
    missing_times = np.argwhere((np.diff(dataset.Date_UTC.values) / 1e9).astype(int) != 60)[:, 0]
    for idx, duration in zip(missing_times, time_diffs[missing_times]):
        start = dates[idx]
        hours = int(np.floor(duration / 3600))
        minutes = int(np.floor((duration - hours * 3600) / 60))
        interval_data = []
        n_nans = []
        for _ in range(max_tries):
            interval_data.append(getDataForInterval(start, hours, minutes))
            n_nans.append(np.isnan(interval_data[-1].to_array().values).sum())
            if n_nans[-1] == 0:
                data_for_interval = interval_data[-1]
                break
        else:
            data_for_interval = interval_data[np.argmin(n_nans)]

        dataset = xr.merge([dataset, data_for_interval])

    if missing_times.shape[0] > 0:
        dataset.to_netcdf(save_fn_pattern.format(year))
